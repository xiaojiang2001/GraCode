from graphviz import Digraph
from sklearn import tree
import graphviz
import sys
import os

def create_network_diagram():
    try:
        dot = Digraph(comment='ResNet with MoE Architecture')
        dot.attr(rankdir='TB')

        # 输入节点
        dot.node('input_vi', 'Visible Input\n(3 channels)')
        dot.node('input_ir', 'Infrared Input\n(3 channels)')

        # ResNetWithMOE 结构
        with dot.subgraph(name='cluster_0') as c:
            c.attr(label='ResNetWithMOE Branch')
            c.node('conv1_vi', 'Conv1\n(16 channels)')
            c.node('moe1', 'MoEAdapter_shallow')
            c.node('conv2_vi', 'Conv2\n(32 channels)')
            c.node('moe2', 'MoEAdapter')
            c.node('conv3_vi', 'Conv3\n(64 channels)')
            c.node('moe3', 'MoEAdapter')
            c.node('conv4_vi', 'Conv4\n(128 channels)')
            c.node('moe4', 'MoEAdapter')

        # 特征融合部分
        dot.node('concat', 'Concatenate')
        dot.node('conv_block4', 'Conv Block 4\n(256->128)')
        dot.node('conv_block5', 'Conv Block 5\n(128->64)')

        # 输出分支
        dot.node('seg_head', 'Segmentation Head')
        dot.node('fusion_head', 'Fusion Head')
        dot.node('decode_vi', 'Decode VI')
        dot.node('decode_ir', 'Decode IR')

        # 添加连接
        # VI 分支
        dot.edge('input_vi', 'conv1_vi')
        dot.edge('conv1_vi', 'moe1')
        dot.edge('moe1', 'conv2_vi')
        dot.edge('conv2_vi', 'moe2')
        dot.edge('moe2', 'conv3_vi')
        dot.edge('conv3_vi', 'moe3')
        dot.edge('moe3', 'conv4_vi')
        dot.edge('conv4_vi', 'moe4')
        dot.edge('moe4', 'concat')

        # IR 分支 (类似的结构)
        dot.edge('input_ir', 'conv1_vi')

        # 融合和输出部分
        dot.edge('concat', 'conv_block4')
        dot.edge('conv_block4', 'conv_block5')
        dot.edge('conv_block5', 'seg_head')
        dot.edge('conv_block5', 'fusion_head')
        dot.edge('conv_block5', 'decode_vi')
        dot.edge('conv_block5', 'decode_ir')
        # 或者方法2：直接显示
        # dot.view()
        # 保存图像
        try:
            # 首先尝试保存为文件
            dot.render('network_structure', format='png', cleanup=True)
            print("图表已保存为 'network_structure.png'")
        except Exception as e:
            print("\n错误：无法生成图表！")
            print("\n请确保您已经正确安装了Graphviz：")
            print("1. 下载并安装Graphviz: https://graphviz.org/download/")
            print("2. 将Graphviz的bin目录添加到系统PATH环境变量中")
            print("   - Windows默认路径通常是: C:\\Program Files\\Graphviz\\bin")
            print("3. 重启您的Python IDE或命令行")
            print("\n详细错误信息:", str(e))
            sys.exit(1)

        return dot  # 添加返回值

    except Exception as e:
        print("\n错误：无法生成图表！")
        print("\n请确保您已经正确安装了Graphviz：")
        print("1. 下载并安装Graphviz: https://graphviz.org/download/")
        print("2. 将Graphviz的bin目录添加到系统PATH环境变量中")
        print("   - Windows默认路径通常是: C:\\Program Files\\Graphviz\\bin")
        print("3. 重启您的Python IDE或命令行")
        print("\n详细错误信息:", str(e))
        sys.exit(1)

if __name__ == '__main__':  # 修复缩进
    create_network_diagram()



