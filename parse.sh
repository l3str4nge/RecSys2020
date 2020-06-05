#!/usr/bin/env bash

RECSYS_DATA_PATH="/data/competitions/recsys2020/Data/" # your recsys data path

mvn compile

GITHASH_STR="$(git log --pretty=format:'%h' -n 1)"
GITCMT_STR="$(git log -1)"
DATE_STR="$(date +"%b_%d_%I:%M%p")"
echo "Date: ${DATE_STR}"
echo "Git: ${GITHASH_STR}"
echo "${GITCMT_STR}"

export MAVEN_OPTS="-Xmx230g -Xms230g"
mvn exec:java -Dexec.mainClass="recsys2020.RecSys20DataParser" -Dexec.args="'${RECSYS_DATA_PATH}'"
echo "Finished parsing in ${RECSYS_DATA_PATH}"