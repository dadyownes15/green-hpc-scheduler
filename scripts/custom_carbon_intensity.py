import csv 
import datetime
import os
def generate_custom_csv(type : str, non_weekday = 450, output_path = "data/"):
    type = type.lower()
    assert type in ["free_weekends"]

    if type == "free_weekends":
        # generate csv for free weekends, meaning that carbon intensities are zero on weekends, and high on weekdays

        field_names = ["MM-DD HH:MM (UTC)", "2021", "2022", "2023"]
        date = datetime.datetime(2021,1,1,0,0)
        csv_path = os.path.join(os.getcwd(), output_path, type + '_minutely.csv')

        with open(csv_path, mode="w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()

            for i in range(8760*60):
                carbon_itensitiy = 0
                day_number = date.day 
                weekday_day_idx =  day_number % 7

                ## Sunday has idx 0
                if weekday_day_idx == 0 or weekday_day_idx == 6:
                    carbon_itensitiy = 450 
                
                row_dict = {
                    "MM-DD HH:MM (UTC)": generate_formatted_date(date),
                    "2021": carbon_itensitiy,
                    "2022": carbon_itensitiy,
                    "2023": carbon_itensitiy,
                }
                writer.writerow(rowdict=row_dict) 
                

                date = generate_next_date(date)



def generate_next_date(prev_date : datetime.datetime):
    timedelta = datetime.timedelta(minutes=1)
    return prev_date + timedelta


def generate_formatted_date(date):
    date_string = str(date)
    date_string.split("-", 1)[1]
    split = date_string.split(":",2)
    date_formatted = split[0] + ":" + split[1]
    return date_formatted




generate_custom_csv(type="free_weekends")