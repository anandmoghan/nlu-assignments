if [ ! -f brown_gutenberg.json ]; then
    echo "Executing the main script"
    python3 main.py
fi
python3 gen_sent.py
