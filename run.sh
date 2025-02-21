#!/bin/bash
echo "Enter w"
read w
gunicorn -w $w -b 0.0.0.0:22 main:app --timeout 0