import streamlit as st

import spacy
from textblob import TextBlob
# from gensim.summarization import summarize

# Sumy Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer('english'))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

def text_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    allData = [('"Tokens": {},\n"Lemma":{}'.format(token.text, token.lemma_)) for token in docx]
    return allData

def entity_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    allData = ['"Tokens": {},\n"Entities":{}'.format(tokens,entities)]
    return allData


def main():
    st.title('NPLiffy with Streanlit')
    st.subheader('Natural Language Processing on the Go')

    # Tokenization
    if st.checkbox('Show Tokens and Lemma'):
        st.subheader('Tokenize Your Text')
        message = st.text_area('Enter Your Text', 'Type Here')
        if st.button('Analyze'):
            nlp_result = text_analyzer(message)
            st.json(nlp_result)

    # Named Entity
    if st.checkbox('Show Named Entities'):
        st.subheader('Tokenize Your Text')
        message = st.text_area('Enter Your Text', 'Type Here')
        if st.button('Analyze'):
            nlp_result = entity_analyzer(message)
            st.json(nlp_result)

    # Sentiment Analysis
    if st.checkbox('Show Sentiment Analysis'):
        st.subheader('Sentiment of Your Text')
        message = st.text_area('Enter Your Text', 'Type Here')
        if st.button('Analyze'):
            blob = TextBlob(message)
            result_sentiment = blob.sentiment
            st.success(result_sentiment)
    # Text Simmarization
    if st.checkbox('Show Text Summarization'):
        st.subheader('Summarize Your Text')
        summary_options = st.select_box('Choose Your Summarizer', 'Type Here')
        if st.button('Summarize'):
            if summary_options == 'gensim':
               st.text('Using Gensim...')
            #    summary_result = summarize(message)
            elif summary_options == 'sumy':
                st.text('Using Sumy...')
                summary_result = sumy_summarizer(message)
            else:
                st.warning('Using Default Summarizer')
                st.text('Using Genism')
                # summary_result = summarize(message)
            
            st.success(summary_result)
    
    st.sidebar.subheader('About The App')
    st.sidebar.text('NLPiffy App with Streankut')
    st.sidebar.info('Kudos to the Streamlit Team')


if __name__ == '__main__':
    main()