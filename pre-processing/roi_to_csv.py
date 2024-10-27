import os
import pandas as pd
from read_roi import read_roi_file

"This is a script to convert ROI files which ends with _measurements to a CSV file."


roi_folder = f""
csv_file = "roi.csv"
match_csv = "match.csv"
data = {
    "original_name": [],
    "train_name": [],
    "roi_file": [],
    "x1": [],
    "x2": [],
    "y1": [],
    "y2": [],
    "length": [],
}

match_df = pd.read_csv(match_csv)
matching_image_to_original = dict(
    zip(match_df["Matching Image Name"], match_df["original"])
)


def main():
    for folder in os.listdir(roi_folder):
        folder_path = os.path.join(roi_folder, folder)
        folder_name = folder.split("_measurements")[0]

        if os.path.isdir(folder_path):

            for i in range(1, 11):
                roi_name = str(i) + ".roi"
                roi_file_path = os.path.join(folder_path, roi_name)
                train_name = matching_image_to_original.get(folder_name, "")

                if os.path.exists(roi_file_path):
                    roi_data = read_roi_file(roi_file_path)

                    for key, value in roi_data.items():
                        y2 = value["y1"]
                        y1 = value["y2"]
                        x1 = value["x1"]
                        x2 = value["x2"]

                        data["original_name"].append(folder_name)
                        data["train_name"].append(train_name)
                        data["roi_file"].append(i)
                        data["x1"].append(x1)
                        data["x2"].append(x2)
                        data["y1"].append(y1)
                        data["y2"].append(y2)
                        data["length"].append(y2 - y1)
                else:
                    data["original_name"].append(folder_name)
                    data["train_name"].append(train_name)
                    data["roi_file"].append(i)
                    data["x1"].append(None)
                    data["x2"].append(None)
                    data["y1"].append(None)
                    data["y2"].append(None)
                    data["length"].append(0)

    df = pd.DataFrame(data)

    df.to_csv(csv_file, index=False)

    print(f"CSV file saved at: {csv_file}")


main()
