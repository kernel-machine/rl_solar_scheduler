DEBIAN_FRONTEND=noninteractive
ln -fs /usr/share/zoneinfo/Europe/Rome /etc/localtime
apt update
apt install -y python3 python3-venv python3-pip htop task-spooler libgl1 libglib2.0-0

