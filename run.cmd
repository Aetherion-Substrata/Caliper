@echo off
if "%CALIPER_MODEL%"=="" echo Set CALIPER_MODEL to your model folder first. && exit /b 1
py -B -W ignore engine.py --serve
