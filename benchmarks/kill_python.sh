#!/usr/bin/env bash
kill -9 $(ps -aux | grep "joblib" | grep -v grep | awk '{printf "%d ", $2}')
kill -9 $(ps -aux | grep "python" | grep -v grep | awk '{printf "%d ", $2}')
