
import matplotlib.pyplot as plt

label_list = ["cascading", "stacking"]    # 各部分标签
size = [68, 2]                       # 各部分大小
color = ["red", "green"]     # 各部分颜色
explode = [0.05, 0]   # 各部分突出值

patches, l_text, p_text = plt.pie(size, explode=explode, colors=color, labels=label_list,
                                  labeldistance=1.1, autopct="%1.1f%%",
                                  shadow=False, startangle=90, pctdistance=0.6)
plt.axis("equal")    # 设置横轴和纵轴大小相等，这样饼才是圆的
plt.legend()
plt.show()
