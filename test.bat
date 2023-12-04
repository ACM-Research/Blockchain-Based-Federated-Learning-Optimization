set "childCount=6"
set "iterations=20"

start "bcfl" run_brownie.bat
ping -n 20 127.0.0.1 > nul
start "bcfl" run_task.bat %childCount% %iterations%
ping -n 8 127.0.0.1 > nul

for /l %%x in (1, 1, %childCount%) do (
    start "bcfl" run_client.bat
)

ping -n 5 127.0.0.1 > nul
start http://localhost:3000

