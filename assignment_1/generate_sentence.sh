if [ ! -f brown_gutenberg.json ]; then
    echo "Executing the main script"
    python main.py
fi
python gen_sent.py
