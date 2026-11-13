№14 зертханалық жұмысты Windows жүйесінде қайта жаңғырту

Жинақтағы README.txt бастапқы орыс жинағының барлық түсіндірмесінің толық
қазақша аудармасын қамтиды. Ондағы 24 бет және баспа тексерулері туралы
жазбалар бастапқы жинақтың тарихи сипаттамасы; қазақша баспа нұсқасының
нақты беттелуі бөлек баспа QA есебімен расталады.

Жинақтың толық көшірмесін қысқа каталогқа орналастырыңыз, мысалы C:\ML\lab14.
Windows жүйесінде ұзын атаулы OneDrive каталогында MLflow FileStore журналы
MAX_PATH шегіне жетіп, WinError 206 беруі мүмкін. Тек скриптті емес,
requirements файлдары мен labs/datasets бумалары бар бүкіл зертханалық
жұмыс жинағын көшіріңіз. PowerShell ішінде:

New-Item -ItemType Directory -Force C:\ML\lab14
Copy-Item -LiteralPath '.\labs' -Destination C:\ML\lab14 -Recurse
Copy-Item -LiteralPath '.\datasets' -Destination C:\ML\lab14 -Recurse
Copy-Item -LiteralPath '.\requirements.txt' -Destination C:\ML\lab14
Copy-Item -LiteralPath '.\requirements-core.txt' -Destination C:\ML\lab14
Set-Location C:\ML\lab14
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python labs/lab14.py

Notebook үшін python -m pip install jupyterlab орындаңыз, jupyter lab іске
қосып, labs/lab14.ipynb ұяшықтарын жоғарыдан төмен орындаңыз. Ядро TCP
127.0.0.1 қосылымына рұқсат беруі керек. HTTP сервері тест соңында жабылады.
MLflow жергілікті файлдық журнал қолданады; сыртқы MLflow сервері қажет емес.
Браузер интерфейсі жеке тексерілген жоқ.

Тексеру журналдарының папкалары verification_artifacts.zip архивінде көрсетілген.
Скрипттің журналы script_run_final, Notebook журналы lab14_artifacts/run
ішінде жасалды. Көшірілген lab14_tracking.json файлындағы архивке қатысты жолдар,
run_id және experiment_id — осы тексеру іске қосуының тарихи метадеректері.
Жаңа таза іске қосуда олар өзгереді. Тұрақты сандық мәндерді
lab14_results.json арқылы салыстырыңыз. Тексеру журналының көшірмесі
verification_artifacts.zip архивінде берілген; жаңа іске қосу өз журналын жасайды.

Манифест пен модельдің SHA-256 мәндері жүктеуге дейін тексеріледі. joblib
артефактісін тек сенімді шығу тегі және сенімді күтілетін хеші болғанда жүктеңіз.
API өріс атаулары, маршруттар, параметрлер, seeds және барлық сандық есептер
бастапқы экспериментпен сақталған. Адам оқитын мәтіндер қазақшаланған.
