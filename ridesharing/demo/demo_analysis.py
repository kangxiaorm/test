from statistic import RidesharingAnalysis, cal_passenger_sharing_time
import pandas as pd
import os
import pickle


def from_pickle(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
        return d


ods = pd.read_pickle('../../data/SanFrancisco_ods.pkl')
requests = ods.copy()
timestamp = requests["pick_up_time"]
requests["timestamp"] = timestamp

# dir_ = os.path.abspath("ridesharing_results_pdtl_wh")  # 根目录
dir_ = os.path.abspath("ridesharing_results_sf")
for paradir in os.listdir(dir_):
    print("processing para: " + paradir)
    sdir_ = os.path.join(dir_, paradir)
    for subdir in os.listdir(sdir_):
        sdir = os.path.join(sdir_, subdir)
        if os.path.isdir(sdir):
            print(f"processing strategy:{subdir}")
            #responding_ratio_df = pd.DataFrame()
            #sharing_ratio_df = pd.DataFrame()
            waiting_time_df = pd.DataFrame()
            addition_time_df = pd.DataFrame()
            #ssdir = os.path.join(sdir, "results")
            ssdir = sdir
            for file in os.listdir(ssdir):
                filename, file_ext = os.path.splitext(file)
                if(file_ext != '.pkl'):
                    continue
                fpath = os.path.join(ssdir, file)
                if os.path.isfile(fpath):
                    print(f"processing file:{file}")
                    date = file[-15:-5]
                    d = from_pickle(fpath)  # every day result
                    pr = d["passenger_records"]
                    if "share_time" not in pr.columns:
                        cal_passenger_sharing_time(pr)
                    reqs = requests[requests["o_date"]
                                    == date]  # transfer date
                    analysis = RidesharingAnalysis(reqs, pr, dir_)
                    #hsr = analysis.hourly_responding_ratio()  # time uit 为时段间隔，例如：60表示以一小时为一个时间段
                    #responding_ratio_df[date] = pd.Series(hsr)
                    #sharing_ratio_df[date] = pd.Series(analysis.hourly_sharing_ratio())
                    waiting_time_df[date] = pd.Series(analysis.hourly_mean_waiting_time())
                    addition_time_df[date] = pd.Series(analysis.hourly_time_addition())
            df = pd.DataFrame({"hourly_waiting_time": waiting_time_df.mean(axis=1),
                               "hourly_addition_time": addition_time_df.mean(axis=1)})
            df.to_csv(os.path.join(sdir, "analysis_result.csv"))
