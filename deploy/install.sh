#!/bin/bash
set +e
SCRIPT=$(readlink -f $0)
SCRIPTPATH=$(dirname ${SCRIPT})
INSTALL_TO="${SCRIPTPATH}"

APP_NAME="trainer"
SERVICE_NAME="${APP_NAME}"
a=`cat /proc/version`
if [[ $a =~ "Red Hat" ]];then
     os=redhat
elif [[ $a =~ Ubuntu ]]; then
     os=ubuntu
elif [[ $a =~ Debian ]]; then
     os=debian
else
     os=debian
fi
if [[ $os == redhat ]]; then
   SERVICE_FILE="/usr/lib/systemd/system/${APP_NAME}.service"
elif [[ $os == ubuntu ]]; then
   SERVICE_FILE="/lib/systemd/system/${APP_NAME}.service"
elif [[ $os == debian ]]; then
   SERVICE_FILE="/lib/systemd/system/${APP_NAME}.service"
fi
SVC_USER=ems
SVC_GROUP=ems

#create svc group and account
function create_svc_account_group(){
    if [[ $(getent group "${SVC_GROUP}") ]]; then
        echo "${SVC_GROUP} group is existed!"
    else
        groupadd -r "${SVC_GROUP}"
    fi

    if [[ $(getent passwd "${SVC_USER}") ]]; then
        echo "${SVC_USER} user is existed!"
    else
        useradd -r -g "${SVC_GROUP}" -s /sbin/nologin "${SVC_USER}"
    fi
    if [ `grep -c "${SVC_USER}" /etc/sudoers` -eq '0' ];then
        sed -i '$a\'"$SVC_USER"' ALL=(ALL:ALL) NOPASSWD: ALL' /etc/sudoers
    fi
}

#systemd script.
function install_as_service() {
    echo "Install ${SERVICE_NAME} as a service"
    echo ${SCRIPTPATH}/${APP_NAME}.service
    sed -i "s:STARTPATH:${INSTALL_TO}:g" ${SCRIPTPATH}/${APP_NAME}.service
    sed -e "s:__INSTALL_TO__:${INSTALL_TO}:g" ${SCRIPTPATH}/${APP_NAME}.service > ${SERVICE_FILE}
    sudo systemctl daemon-reload
    sudo systemctl enable ${SERVICE_NAME}
    sudo systemctl start ${SERVICE_NAME}
}

function check_service() {
    if [[ -f ${SERVICE_FILE} ]]; then
        echo "Service \"${SERVICE_NAME}\" already existed."
        return 1
    fi
}

#create log dir if no
function create_log_dir() {
    local log_dir="$(dirname ${SCRIPTPATH})/logs"
    if [[ ! -d "${log_dir}" ]]; then
        mkdir "${log_dir}"
    fi
    if [ ! -f "${log_dir}/gateway.log" ];then
            touch "${log_dir}/${APP_NAME}.log"
    fi
}

function main() {
    echo "installing ..."
    if check_service; then
        install_as_service
    else
        echo "${SERVICE_NAME} service already installed"
        rm -rf ${SERVICE_FILE}
        install_as_service
    fi

    create_svc_account_group
    #create_log_dir
    chown -R "${SVC_USER}":"${SVC_GROUP}" "$(dirname ${INSTALL_TO})"

    systemctl daemon-reload
    systemctl enable ${APP_NAME}
    systemctl start ${APP_NAME}
}

main "@"

