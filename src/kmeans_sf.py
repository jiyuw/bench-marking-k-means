'''
-- snowflake format
-- load data into snowflake
CREATE TABLE datapoints AS (
    coord ARRAY
);

CREATE TABLE centroids AS (
    id INT
    coord ARRAY
);
'''
import numpy as np
from tqdm.notebook import trange
import pandas as pd

def load_dataset_sf(con, file):
    """
    load s3 file into a table
    :param con: snowflake connector
    :param file: s3 path
    :param table: table name to put in data
    :return:
    """
    if '-' not in file:
        table = 'datapoints'
        cquery = f"CREATE OR REPLACE TABLE {table} (coord string);"
    elif '-ic.txt' in file:
        table = 'centroids'
        cquery = f"CREATE OR REPLACE TABLE {table} (id INT NOT NULL IDENTITY(1,1), coord string);"
    else:
        return None
    con.execute('use schema python_516.public')

    # create table
    con.execute(cquery)

    # stage file
    con.execute(
        f"create or replace stage s3_stage storage_integration = connectS3 url = '{file}' file_format = python516;")

    # copy data into table
    con.execute(f"TRUNCATE TABLE IF EXISTS {table}")
    con.execute(f"COPY INTO {table} (coord) FROM @s3_stage")

    string_to_float = "CREATE OR REPLACE FUNCTION dist_to_num(a ARRAY) RETURNS ARRAY LANGUAGE JAVASCRIPT AS $$ " \
                      "return A.map(Number)$$"
    con.execute(string_to_float)

    if table == 'datapoints':
        con.execute("CREATE OR REPLACE TABLE datapoints AS "
                    "SELECT dist_to_num(strtok_to_array(t.coord, ' ')) coord "
                    "FROM datapoints AS t")
    else:
        con.execute("CREATE OR REPLACE TABLE centroids AS "
                    "SELECT dist_to_num(strtok_to_array(t.coord, ' ')) coord, t.id id "
                    "FROM centroids AS t")

    return table


def kmeans_sf(con, X, cen, max_iter):
    if X != 'datapoints':
        raise ValueError("wrong name for X")
    if cen != 'centroids':
        raise ValueError("wrong name for centroids")

    # create a udf to calculate distance between two points
    find_id_udf = "CREATE OR REPLACE FUNCTION find_id(a ARRAY, cen ARRAY) " \
                  "RETURNS TABLE(id FLOAT) " \
                  "LANGUAGE JAVASCRIPT AS '{ " \
                  "processRow: function(row, context){ " \
                  "var n = row.A.length; " \
                  "var m = row.CEN.length; " \
                  "for(var i=0;i<m;i++){ " \
                  "var d = 0; " \
                  "for(var j=0;j<n;j++){ " \
                  "d = d+Math.pow((row.A[j]-row.CEN[i][j]),2) " \
                  "} " \
                  "if(i==0){this.id=1;this.dist=d} " \
                  "else if(d<this.dist){this.id=i+1;this.dist = d} " \
                  "} " \
                  "}, " \
                  "finalize: function(rowWriter, context){ " \
                  "rowWriter.writeRow({ID:this.id}) " \
                  "} " \
                  "}';"
    con.execute(find_id_udf)

    # array length
    con.execute(f"SELECT ARRAY_SIZE(coord) FROM centroids LIMIT 1")
    n = con.fetchone()[0]

    new_cen_udf = "CREATE OR REPLACE FUNCTION array_average(id FLOAT, p ARRAY) " \
                  "RETURNS TABLE(id FLOAT, c ARRAY) " \
                  "LANGUAGE JAVASCRIPT AS '{" \
                  "processRow: function (row, context) {" \
                  " this.ccount = this.ccount + 1;" \
                  " this.id = row.ID;" \
                  " for(var i=0;i<" + str(n) + ";i++){" \
                  "  this.csum[i] = this.csum[i]+row.P[i]};" \
                  "}," \
                  "finalize: function (rowWriter, context) {" \
                  " for(var i=0;i<" + str(n) + ";i++){" \
                  "  this.csum[i] = this.csum[i]/this.ccount}" \
                  " rowWriter.writeRow({ID: this.id, C:this.csum})" \
                  "}," \
                  "initialize: function (argumentInfo, context) {" \
                  " this.ccount = 0;" \
                  " this.id = 0;" \
                  " this.csum = new Array(" + str(n) + ").fill(0);" \
             "}" \
             "}'"
    con.execute(new_cen_udf)

    for i in trange(max_iter):
        # obtain a table with assignment for each point
        closest_cen = "CREATE OR REPLACE TABLE closest_cen AS " \
                      "SELECT d.coord coord, t.id id " \
                      "FROM (SELECT dp.coord coord, (SELECT array_agg(coord) FROM centroids ORDER BY id) cen " \
                      "FROM datapoints dp) d, " \
                      "TABLE(find_id(d.coord, d.cen) OVER (PARTITION BY d.coord)) t"
        con.execute(closest_cen)

        if i == max_iter - 1:
            break

        # assign new centroids and update the cen table
        cmd_update_cen = "CREATE OR REPLACE TABLE centroids AS " \
                         "SELECT t.id id, t.c coord " \
                         "FROM closest_cen, TABLE(array_average(id, coord) OVER (PARTITION BY id)) t"
        con.execute(cmd_update_cen)

    con.execute("SELECT * FROM closest_cen")
    final_assign = con.fetch_pandas_all()

    con.execute("SELECT * FROM centroids")
    final_cen = con.fetch_pandas_all()

    # formating
    final_assign['COORD'] = final_assign['COORD'].apply(
        lambda row: np.array(row.replace('[', '').replace(']', '').replace('\n', '').split(','), dtype=float))
    final_cen['COORD'] = final_cen['COORD'].apply(
        lambda row: np.array(row.replace('[', '').replace(']', '').replace('\n', '').split(','), dtype=float))

    tmp_assign = pd.DataFrame(final_assign.COORD.values.tolist(), final_assign.index).add_prefix('x_')
    final_assign = np.array(pd.concat([final_assign, tmp_assign], axis=1, sort=False).drop(columns='COORD'))

    tmp_cen = pd.DataFrame(final_cen.COORD.values.tolist(), final_cen.index).add_prefix('x_')
    final_cen = np.array(pd.concat([final_cen, tmp_cen], axis=1, sort=False).drop(columns='COORD'))

    return final_assign, final_cen, None
