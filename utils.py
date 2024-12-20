import logging



def read_session_info(session_path):
    
    data = {
        "list_fc1_units": [],
        "list_fc2_units": [],
        "list_eps_start": [],
        "list_episodes": [],
        "numb_of_trains": None
    }
    with open(session_path, "r") as file:
        lines = file.readlines()
        
        for line in lines:
            if "list_fc1_units" in line:
                data["list_fc1_units"] = eval(line.split(":")[1].strip())
            elif "list_fc2_units" in line:
                data["list_fc2_units"] = eval(line.split(":")[1].strip())
            elif "list_eps_start" in line:
                data["list_eps_start"] = eval(line.split(":")[1].strip())
            elif "list_episodes" in line:
                data["list_episodes"] = eval(line.split(":")[1].strip())
            elif "numb_of_trains" in line:
                data["numb_of_trains"] = int(line.split(":")[1].strip())
    
    return data

# # 从日志中读取数据
# loaded_data = read_log_file("data_log.txt")

# # 恢复变量
# list_fc1_units = loaded_data["list_fc1_units"]
# list_fc2_units = loaded_data["list_fc2_units"]
# list_eps_start = loaded_data["list_eps_start"]
# list_episodes = loaded_data["list_episodes"]
# numb_of_trains = loaded_data["numb_of_trains"]


def write_session_info(session_path,data):

    # 配置日志
    logging.basicConfig(filename=session_path, level=logging.INFO, format="%(asctime)s - %(message)s")

    # 数据
    list_fc1_units = data["list_fc1_units"]
    list_fc2_units = data["list_fc2_units"]
    list_eps_start = data["list_eps_start"]
    list_episodes = data["list_episodes"]
    numb_of_trains = data["numb_of_trains"]  # 10
    # 记录数据到日志
    logging.info(f"list_fc1_units: {str(list_fc1_units)}")
    logging.info(f"list_fc2_units: {str(list_fc2_units)}")
    logging.info(f"list_eps_start: {str(list_eps_start)}")
    logging.info(f"list_episodes: {str(list_episodes)}")
    logging.info(f"numb_of_trains: {str(numb_of_trains)}")
    logging.shutdown()
    print(f"数据已保存到日志文件{session_path}")