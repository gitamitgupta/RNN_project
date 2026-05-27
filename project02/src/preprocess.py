import json

def load_data(path, limit=5000):

    headlines = []

    with open(path, 'r', encoding='utf-8') as file:

        for i, line in enumerate(file):

            if i >= limit:
                break

            data = json.loads(line)

            headlines.append(data['headline'])

    return headlines