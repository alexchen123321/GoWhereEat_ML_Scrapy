import json


def filter():
    with open('chop.json') as f:
        data = json.load(f)
    # print(data)
    # print(data[1]["name"])
    output_dict = [x for x in data if ['name'] != [] ]
    out_json = json.dumps(output_dict,ensure_ascii=False)
    print(out_json)
    with open('new_chop_data.json', 'w') as outfile:
        outfile.write(out_json)

# filter()
# with open('new_chop_data.json') as f:
#     data = json.load(f)
#     print(data[0])
