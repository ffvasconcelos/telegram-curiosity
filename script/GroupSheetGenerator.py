import numpy as np
import pandas as pd
import json
import psycopg2
import time
from tqdm import tqdm


def make_data_object(dataframe):
    data_object = {}

    for i in dataframe.index:
        data_object[dataframe['id'][i]] = json.loads(dataframe['data'][i])
        # aux = json.loads(dataframe['data'][i])
        # data_object[dataframe['id'][i]] = json.loads(aux)

    return data_object


def group_formatter(df):
    data_object = make_data_object(df)

    uuids = []
    message_ids = []
    group_ids = []
    user_ids = []
    message_categories = []
    dates = []

    for i in tqdm(data_object):
        try:
            if data_object[i]['id'] is not None \
            and data_object[i]['peer_id']['channel_id'] is not None \
            and data_object[i]['from_id']['user_id'] is not None:
                uuids.append(i)
                message_ids.append(str(data_object[i]['id']))
                group_ids.append(str(data_object[i]['peer_id']['channel_id']))
                user_ids.append(str(data_object[i]['from_id']['user_id']))
                dates.append(data_object[i]['date'])

                if data_object[i]['media'] is None:
                    message_categories.append(0)
                elif data_object[i]['media']['_'] == 'MessageMediaDocument':
                    message_categories.append(3)
                elif data_object[i]['media']['_'] == 'MessageMediaPhoto':
                    message_categories.append(2)
                elif data_object[i]['media']['_'] == 'MessageMediaWebPage':
                    message_categories.append(1)
                elif data_object[i]['media']['_'] == 'MessageMediaUnsupported':
                    message_categories.append(5)
                else:
                    message_categories.append(4)
        except Exception as e:
            print('Data object error')
            print(i)
            print('---------------------------------------------------------------')

    return pd.DataFrame(data={
        'uuid': uuids,
        'message_id': message_ids,
        'groupId': group_ids,
        'groupname': group_ids,
        'username': user_ids,
        'usernumber': user_ids,
        'message_category': message_categories,
        'timestamp': dates
    })


if __name__ == "__main__":
    start = time.time()
    conn = psycopg2.connect(
        host='192.168.2.19',
        # host='localhost',
        database='telegram_db',
        user='postgres',
        password='1234'
    )

    cur = conn.cursor()

    cur.execute("(SELECT m.channel_id, COUNT(*) AS quantidade FROM messages m WHERE m.channel_id IN (SELECT "
                "channel_id FROM messages GROUP BY channel_id HAVING COUNT(DISTINCT from_id) > 1) GROUP BY "
                "m.channel_id) ORDER BY quantidade")

    recset = cur.fetchall()

    groups = np.array([rec[0] for rec in recset])

    group_names = pd.DataFrame(groups, columns=['gname'])
    group_names.to_csv('./dataset_gname_raw_info.csv', sep='\t', index=False)

    for group in groups:
        query = "SELECT t1.* FROM messages t1 WHERE t1.channel_id = %d" % group
        cur.execute(query)

        group_df = pd.DataFrame(cur.fetchall(),
                                columns=["id", "message_id", "channel_id", "data", "retrieved_utc", "updated_utc",
                                         "message", "views", "forwards", "from_id", "post_author", "fwd_from_id",
                                         "fwd_from_name", "fwd_post_author", "message_utc"])
        group_df = group_df.drop(
            columns=['views', 'forwards', 'fwd_from_id', 'fwd_from_name', 'fwd_post_author', 'post_author'])

        print("Formating group %s sheet" % str(group))
        formatted_group_df = group_formatter(group_df)
        formatted_group_df.to_csv('./groups/' + str(group) + '.tsv', sep='\t', index=False)

    print("Code ran for %f seg" % ((time.time() - start)))

    conn.close()
