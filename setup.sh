mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"husayn.ali916@sci.s-mu.edu.eg\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
