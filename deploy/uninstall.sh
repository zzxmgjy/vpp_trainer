#!/bin/bash
echo "<1>-stop trainer service -----"
systemctl stop trainer.service
systemctl daemon-reload
echo "<2>-disable trainer service -----"
systemctl disable trainer.service
echo "<3>-del trainer service file -----"
rm -rf /usr/lib/systemd/system/trainer.service
rm -rf /lib/systemd/system/trainer.service
echo "<done>-----"
