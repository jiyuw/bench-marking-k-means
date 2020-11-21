'''
-- snowflake format
-- load data into snowflake
CREATE TABLE datapoints AS (
    x INTEGER,
    y INTEGER
);

CREATE TABLE centroids AS (
    cen_x INTEGER,
    cen_y INTEGER
);

stage_file_format = (TYPE = 'txt' FIELD_DELIMITER=' ');

-- load local file to stage area then table
PUT file://D:\ @%datapoints

COPY INTO datapoints;


-- kmeans functions
'''

import numpy as np


def load_dataset_sf(con, file, table):
    """
    load local file into a table
    :param con: snowflake connector
    :param file: absolute path to the file
    :param table: table name to put in data
    :return:
    """
    con.cursor().execute(f"TRUNCATE TABLE IF EXISTS {table}")
    con.cursor().execute(f"PUT file://{file} @%{table}")
    con.cursor().execute(f"COPY INTO {table}")


def kmeans_sf(con, X, cen, max_iter):
    cost_output = []

    n_iter = 0
    while n_iter < max_iter:
        # cross join X table and cen table
        con.cursor().execute(f"CREATE OR REPLACE TABLE data_cen AS SELECT * FROM {X} CROSS JOIN {cen}")

        # calculate data to cen euclidean distance
        cmd_data_cen = "CREATE OR REPLACE TABLE data_close_cen AS" \
                       "WITH dc_dist AS (SELECT x, y, cen_x, cen_y, ((x-cen_x)**2+(y-cen_y)**2) AS dist FROM data_cen)" \
                       "SELECT a.x, a.y, b.cen_x, b.cen_y, a.min_dist AS min_dist" \
                       "FROM (SELECT x, y, MIN(dist) AS min_dist FROM dc_dist GROUP BY x, y) AS a, dc_dist AS b" \
                       "WHERE a.x = b.x AND a.y = b.y AND a.min_dist = b.dist"
        con.cursor().execute(cmd_data_cen)
        cost = con.cursor().execute("SELECT SUM(min_dist) FROM data_close_cen")
        cost_output.append(cost)

        if n_iter == max_iter-1:
            break

        # assign new centroids and update the cen table
        cmd_update_cen = f"UPDATE {cen}" \
                         f"SET cen_x = AVG(a.x), cen_y = AVG(a.y)" \
                         f"FROM data_close_cen AS a" \
                         f"GROUP BY cen_x, cen_y"
        con.cursor().execute(cmd_update_cen)

        n_iter += 1

    data_sql = "SELECT * FROM data_close_cen"
    con.cursor().execute(data_sql)
    final_assign = con.cursor().fetch_pandas_all()

    cen_sql = "SELECT * FROM cen"
    con.cursor().execute(cen_sql)
    final_cen = con.cursor().fetch_pandas_all()

    return np.array(final_cen), np.array(final_assign), cost_output
