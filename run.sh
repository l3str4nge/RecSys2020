#!/usr/bin/env bash

RECSYS_PATH="/data/competitions/recsys2020/" # your recsys path
HIGHL2="false" # high L2 run?

mvn compile

GITHASH_STR="$(git log --pretty=format:'%h' -n 1)"
GITCMT_STR="$(git log -1)"
DATE_STR="$(date +"%b_%d_%I:%M%p")"
echo "Date: ${DATE_STR}"
echo "Git: ${GITHASH_STR}"
echo "${GITCMT_STR}"

TARGET_NAME="LIKE"
echo "Running ${TARGET_NAME}"
export MAVEN_OPTS="-Xmx230g -Xms230g"
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20Model" -Dexec.args="'trainLIBSVM:${TARGET_NAME}' '${RECSYS_PATH}'"
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20Model" -Dexec.args="'trainXGB:${TARGET_NAME}:${HIGHL2}' '${RECSYS_PATH}'"
export MAVEN_OPTS="-Xmx220g -Xms220g"
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20Model" -Dexec.args="'submit:${TARGET_NAME}' '${RECSYS_PATH}'"
echo "Date: ${DATE_STR}"
echo "Git: ${GITHASH_STR}"
echo "${GITCMT_STR}"