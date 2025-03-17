:: Activate the virtual environment
call venv\Scripts\activate

:: Run the app in the background
echo Running app...
start python web_app/app.py

:: Open the app in the browser
start "" http://127.0.0.1:8050

:: Keep the window open after the app runs
echo Press any key to exit...
pause
