import csv
class DatabaseHandler:
    def __init__(self, file_path):
        self.file_path = file_path

    def store_encodings(self, data):
        with open(self.file_path, 'a', newline='') as csvfile:
            fieldnames = ['Identifier', 'Encoding']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for identifier, encoding in data:
                writer.writerow({'Identifier': identifier, 'Encoding': encoding})

    def load_encodings(self):
        data = {}
        try:
            with open(self.file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    identifier = row['Identifier']
                    encoding = row['Encoding']
                    data[identifier] = encoding
        except FileNotFoundError:
            pass  # Handle the case when the file doesn't exist

        return data
