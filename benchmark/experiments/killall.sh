eai job ls --me --state alive --fields id,name | grep starcaster | cut -d ' ' -f1 | xargs eai job kill || echo "Nothing to kill."
