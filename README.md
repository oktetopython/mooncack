# mooncack
okteto Desktop


##配置

A. 在Linux VPS的Terminal上安装配置命令：

1. 安装npm命令：

apt install npm

2. 安装wstunnel命令：

sudo npm install -g wstunnel
--------------------------提前准备----------------------------------
###可能有用
4. 项目安装成功后本地终端进行配置：

wstunnel -t 1234:127.0.0.1:22 wss://你在okteto上新建项目的网址/ &



5. 本地终端继续ssh获取管理员root权限：

ssh root@127.0.0.1 -p 3333
6. 安装VNC：

apt install tightvncserver

7. 安装Linux桌面：

apt install lxde

8．配置VNC远程登录密码：

vncserver :1

9. 保持第一个终端状态不变，不要关闭第一个终端窗口，开第二个终端窗口并输入：

wstunnel -t 5806:127.0.0.1:5901 wss://你在okteto云上的项目网址/ &

例子：wstunnel -t 5806:127.0.0.1:5901 wss://areyouok-uncleluo.cloud.okteto.net/ &

10. 本地VNC Viewer客户端登录：

127.0.0.1:5806
