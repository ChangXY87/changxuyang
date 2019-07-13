import requests
import re
from bs4 import BeautifulSoup
import requests
import math
import networkx as nx
import matplotlib.pyplot as plt

#爬取数据
# headers =  {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.221 Safari/537.36 SE 2.X MetaSr 1.0'}
# url = 'http://ditie.mapbar.com/beijing/'
# start_html = requests.get(url, headers=headers)
# soup = BeautifulSoup(start_html.text, 'lxml')
# all_a = soup.find('div',class_='ditie_news ditie_color').p.find_all('a')
#
# #获取站点数据
# line_station_dict = {}
# for a in all_a:
#     all_a2_list = []
#     line_name = a.get_text()
#     #print(line_name)
#     station_url = a['href']
#     second_html = requests.get(station_url, headers=headers)
#     soup_2 = BeautifulSoup(second_html.text, 'lxml')
#     all_a2 = soup_2.find_all('a', class_='cl-station')
#     all_a2_list.append(all_a2)
#     station_list = []
#     for a in all_a2_list:
#         for n in a:
#             station = n.get_text()
#             station_list.append(station)
#     #print(station_list)
#     line_station_dict[line_name] = station_list
#
# #保存站点数据
# def text_save(content, filename, mode='a'):
#     file = open(filename, mode,encoding='utf-8')
#     file.write(str(content))
#     file.close()
# # text_save(line_station_dict,'line_station_dict.txt')
# # text_save(station_list,'station_llist.txt')

#读取网页爬取的数据
station_all = open('station_llist.txt', 'r',encoding='utf-8')
stationinfo = station_all.read()
lines_all = open('line_station_dict.txt', 'r',encoding='utf-8')
lines = eval(lines_all.read())

#得到各站点间的连接图
def get_station_connect(lines):
    stations = set()
    for key in lines.keys():
        stations.update(set(lines[key]))
    system = {}
    for station in stations:
        next_station = []
        for key in lines:
            if station in lines[key]:
                line = lines[key]
                idx = line.index(station)
                if idx == 0:
                    next_station = line[1]
                elif idx == len(line)-1:
                    next_station = line[idx-1]
                else:
                    next_station = [line[idx-1]]
                    next_station.append(line[idx+1])
        system[station] = next_station
    return system

station_location = {}
for line in stationinfo.split('\n'):
    station_name = re.findall('(\w+)',line)[0]
    x_y = re.findall('\w+,(\d+.\d+),(\d+.\d+)',line)[0]
    x_y = tuple(map(float,x_y))
    station_location[station_name] = x_y

#得到两站点间距离
def geo_distance(origin,destination):
    lat1,lon1 = origin
    lat2,lon2 = destination
    radius = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d

def get_station_distance(station1,station2):
    return geo_distance(station_location[station1],station_location[station2])

#作图(可以标记节点，但作边时报错)
# stations = list(station_location.keys())
# station_graph = nx.Graph()
# station_graph.add_nodes_from(stations)
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.figsize'] = (30,10)
#
station_connect = get_station_connect(lines)
# station_connect_graph = nx.Graph(station_connect)
# station_image = nx.draw(station_graph,station_location,with_labels=True,node_size=10,font_size=8)
#plt.show()

print(station_connect)

#定义搜索函数
def search(start,destination,connection_graph,sort_candidate):
    pathes = [[start]]
    visited = set()
    while pathes:
        path = pathes.pop(0)
        froninter = path[-1]
        if froninter in visited: continue
        successors = connection_graph[froninter]
        for station in successors:
            if station in path:continue
            new_path = path + [station]
            pathes.append(new_path)
            if station == destination: return new_path
        visited.add(froninter)
        pathes = sort_candidate(pathes)

#最短站点数
def transfer_less_first(pathes):
    return sorted(pathes,key=len)

#（报错）
path = search('四惠东站','西直门站',station_connect,sort_candidate=transfer_less_first)
print(path)