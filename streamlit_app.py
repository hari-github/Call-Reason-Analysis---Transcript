import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import os
import base64
import re
from openai import AzureOpenAI
import asyncio
import sys

# if sys.platform.startswith('win') and sys.version_info >= (3, 8):
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ["STREAMLIT_SERVER_ENABLE_TELEMETRY"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

st.set_page_config(layout="wide", page_title="Call Reasons Analysis")

st.title("Call Center Analytics - Call Reasons Analysis")
# st.write("Upload an Excel file to visualize data with interactive charts")

api_key = st.text_input("Enter the OpenAI API Key:")
st.write(f"You entered: {api_key}")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])

if 'df' not in st.session_state:
    st.session_state.df = None

def process_data(file):
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None
    
    
def client_chat(api_key) :

    
    # Initialize Azure OpenAI Service client with key-based authentication
    
    endpoint = os.getenv("ENDPOINT_URL", "https://harip-m9uh421i-eastus2.openai.azure.com/")
    
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2025-01-01-preview",
    )
    
    return client

def chat(msg, client) :
    
    
    deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
    subscription_key = "AZURE_OPENAI_API_KEY"

    completion = client.chat.completions.create(
        model=deployment,
        messages=msg,
        # max_tokens=800,
        temperature=0,
        top_p=0.01,
        # frequency_penalty=0,
        # presence_penalty=0,
        # stop=None,
        stream=False
    )

    return (completion.choices[0].message.content)

def topic(i, dat, client) :
  conv = dat.text[i]
  # print (dat.topic[i])
  # topic_li = list(set(topic_repositary))
  system_msg = """You are going to act as a Topic Model. Your job is to analyse the call transcripts between agent and member of a health insurance
                company and determine the key topics that are discussed in the conversation and return the topics.

                You need to look for topics related to health insurance that are discussed in the conversation. You need to give a sentiment of Positive, Negative & Neutral for each topic discussed in the conversation based on
                the member experience. Also, include a summary of the call capturing the member emotions.
                """
  user_msg = f"""Determine the topic and sentiment for this conversation - {conv}
                Response should be in this format <Output>{{"Topic1" : "<Topic1>", "Sentiment1" : "<Sentiment>", "Topic2" : "<Topic2>", "Sentiment2" : "<Sentiment>", "Topic3" : "<Topic3>", "Sentiment3" : "<Sentiment>", "Summary" : "<Summary>"}}</Output>
                Look for a maximum of 2 or 3 key important topics discussed.
                Topic should be related to general health insurance areas like appointment, bills, payment related, claims, medications, benefits, etc.
              """
  msg = [ {"role": "system", "content": system_msg},
              {"role": "user", "content": user_msg}
              ]
  res = chat(msg, client)
  return res

def topic_extract(dat, client):
    res_list = []
    topic_list = []
    for i in range(len(dat)) :
      res = topic(i, dat, client)
    
      results = re.findall(r'<Output>(.*?)</Output>', res, re.DOTALL)
      res_list.append(results)
      # print (results)
      print (i)
      topic_val = []
      topic1 = eval(results[0])['Topic1'] + " - " + eval(results[0])['Sentiment1']
      # topic_repositary.append(eval(results[0])['Topic1'])
      topic_val.append(topic1)
      try :
        topic2 = eval(results[0])['Topic2'] + " - " + eval(results[0])['Sentiment2']
        # topic_repositary.append(eval(results[0])['Topic2'])
        topic_val.append(topic2)
      except :
        topic2 = ''
      try :
        topic3 = eval(results[0])['Topic3'] + " - " + eval(results[0])['Sentiment3']
        # topic_repositary.append(eval(results[0])['Topic3'])
        topic_val.append(topic3)
      except :
        topic3 = ''
        pass
      # summary = eval(results[0])['Summary']
      # print (summary)
      topic_list.append(topic_val)
      # print (topic_repositary)
      
    return res_list, topic_list
  
def cluster(new_topic, n, client) :

  # topic_li = list(set(topic_repositary))
  system_msg = f"""You are going to act as a topic correction tool. Your job is to analyse the topics and create common headers. I have a lot of topics that have values
                that are small variant of each other. I need to create topics that cover all the variants and return a common topic. Topics are determined from the call transcript
                of health insurance company.

                New headers should also be based on the health insurance customer service topics. Number of headers should not be more than {n}. Optimize the headers based on the 
                themes and assign them to the topics.
                """
  user_msg = f"""You need to analyse the topics and create a header for each topic. 

                Steps to create headers -
                1. This excercise is similar to clustering
                2. Read through the entire topic list
                3. Create upto {n} themes that covers all the topics from the list
                4. Assign the newly determined themes back to the topic list 
                Here is the Topic list - {new_topic}
                Response should be in this format <Output>{{"Topic1" : "<theme>", "Topic2" : "<theme>", "Topic3" : "<theme>", "Topic4" : "<theme>"}}</Output>
              """
  msg = [ {"role": "system", "content": system_msg},
              {"role": "user", "content": user_msg}
              ]
  res = chat(msg, client)
  return res

# dat['topic_val'] = topic_list

def cluster_process(dat,n, client) :
        df = dat.explode('topic_val')
        df['New_topic'] = df['topic_val'].apply(lambda x: x.split(' - ')[0])
        
        new_topic = (df['New_topic'].unique())
        
        res = cluster(new_topic,n, client)
            
        results = re.findall(r'<Output>(.*?)</Output>', res, re.DOTALL)
        x = eval(results[0])
        old = list(x.keys())
        new = list(x.values())
        
        lu = pd.DataFrame({'old': old, 'new': new})
        df = df.merge(lu, left_on='New_topic', right_on='old', how='left')
        df['Summary'] = df['res'].apply(lambda x: eval(x[0])['Summary'])
        df['transcript'] = df['text']
        df['topic'] = df['new']
        df['gran_topic'] = df['New_topic']
        df['Sentiment'] = df['topic_val'].apply(lambda x: x.split(' - ')[1])
        
        df_final = df[['transcript', 'topic', 'gran_topic', 'Sentiment', 'Summary']]
        return df_final

def clean(new_topic, client) :

  # topic_li = list(set(topic_repositary))
  system_msg = f"""You are going to act as a topic correction tool. Your job is to analyse the topics and combine topics that are semantically same.  
                I have a lot of topics that have values that are small variant of each other. You need to create topics that cover all the variants and return a common topic. 
                Topics are determined from the call transcript of health insurance company. Do not try to cluster too many topics that are far different from each other.

                New headers should also be based on the health insurance customer service topics. Primary thing to look for are plural/singlular, same topic represented differently.
                """
  user_msg = f"""You need to analyse the topics and create a cleaner version for each topic. 

                Steps to create headers -
                1. This excercise is not a clustering excercise but more like a topic correction excercise
                2. Read through the entire topic list
                3. Look for same topics represented differently and create a common topic for them. Do not combine topics that are semantically not same
                3. You need to retain the granularity of information by not creating themes. Make sure you are not dissolving the information from the original topic. 
                    Retain as much infomation as possible. You can create as many new topics as you want.
                4. Assign the newly determined themes back to the topic list 
                5. If any topic is unique, retain the same topic
                Here is the Topic list - {new_topic}
                Response should be in this format <Output>{{"Topic1" : "<theme>", "Topic2" : "<theme>", "Topic3" : "<theme>", "Topic4" : "<theme>"}}</Output>
              """
  msg = [ {"role": "system", "content": system_msg},
              {"role": "user", "content": user_msg}
              ]
  res = chat(msg, client)
  return res

def clean_process(dat, client) :
    
    new_topic = (dat['gran_topic'].unique())
    res = clean(new_topic, client)
    results = re.findall(r'<Output>(.*?)</Output>', res, re.DOTALL)
    x = eval(results[0])
    old = list(x.keys())
    new = list(x.values())
    
    lu = pd.DataFrame({'old': old, 'new': new})
    df = dat.merge(lu, left_on='gran_topic', right_on='old', how='left')
    df['Subtopic'] = df['new']

    df = df[['transcript', 'topic', 'gran_topic', 'Subtopic', 'Sentiment', 'Summary']]
    
    return df


    

if uploaded_file is not None and st.button("Analyse"):
    # Read data
    dat = process_data(uploaded_file)
    
    if dat is not None:
        st.write("### Data Preview")
        st.dataframe(dat.head())
        
        # Get columns for selection
        columns = dat.columns.tolist()
        
        
        
        
        # value = st.number_input("Enter the number of topics:")

        # st.write(f"You entered: {value}")
        
        df = st.session_state.get("df", None)
        
        client = client_chat(api_key)
        
        dat['res'], dat['topic_val'] = topic_extract(dat,client)
        
        # if value :
        #     n = value
        #     df_n = cluster_process(dat,n)
        # else :
        #     n = 15
        #     df_n = cluster_process(dat,n)
        
        n = 15
        df_n = cluster_process(dat,n, client)
       
            
        df = clean_process(df_n, client)
        
        st.session_state.df = df
        st.success("File processed successfully.")
        

df = st.session_state.get("df", None)

if df is not None :        
        
    tab1, tab2 = st.tabs(["📈 Charts", "🧮 Data Preview"])
    
    with tab1 :
    
    
        
        # Select topic column
        # topic_col = st.selectbox("Select topic column for primary chart", columns)
        topic_col = 'topic'
        # Select header column for secondary chart
        # header_col = st.selectbox("Select header column for secondary chart", columns)
        header_col = 'Subtopic'
        sent_col = 'Sentiment'
        
        if topic_col and header_col:
            # Create main container for charts
            chart_container = st.container()
            
            with chart_container:
                col1, drop1 = st.columns(2)
                
                with col1:
                    st.write(f"### Topic Discussed in the Call")
                    
                    # Calculate value counts and create primary chart
                    topic_counts = df[topic_col].value_counts().reset_index()
                    topic_counts.columns = [topic_col, 'Count']
                    
                    fig1 = px.bar(
                        topic_counts, 
                        x=topic_col, 
                        y='Count',
                        color=topic_col,
                        title=f"Distribution by {topic_col}"
                    )
                    
                    # Configure layout
                    fig1.update_layout(height=500)
                    
                    # Render the chart
                    st.plotly_chart(fig1, use_container_width=True)
                    
                with drop1:
                    
                    # Get unique topic values for dropdown
                    unique_topics = df[topic_col].unique().tolist()
                    
                    # Add dropdown for topic selection
                    selected_topic = st.selectbox(
                        f"Select a {topic_col} value to explore:", 
                        options=unique_topics
                    )
                    
                    # Add button to generate secondary chart
                    generate_button = st.button("Generate Secondary Chart")
                    
                    filtered_df = df[df[topic_col] == selected_topic]
                    
                col2, col3 = st.columns(2)
                
    
                with col2:
                    # Show secondary chart based on selection and button click
                    if generate_button:
                        st.write(f"### {header_col} Distribution for {selected_topic}")
                        
                        # Filter data for the selected topic
                        filtered_df = df[df[topic_col] == selected_topic]
                        
                        if not filtered_df.empty:
                            # Create count by header for filtered data
                            header_counts = filtered_df[header_col].value_counts().reset_index()
                            header_counts.columns = [header_col, 'Count']
                            
                            fig2 = px.bar(
                                header_counts,
                                x=header_col,
                                y='Count',
                                title=f"{header_col} Distribution for {selected_topic}",
                                color=header_col
                            )
                            
                            fig2.update_layout(height=500)
                            st.plotly_chart(fig2, use_container_width=True)
                            
                
                        else:
                            st.info(f"No data found for {selected_topic}")
                    else:
                        st.info("Select a topic and click 'Generate Secondary Chart' to view details")
                
                # with tab1 :
                    
                #     if not filtered_df.empty:
                    
                #         # Show filtered data
                #         with st.expander(f"Show data for {selected_topic}"):
                #             st.dataframe(filtered_df[['Summary','Subtopics','Sentiment']])
                            
                # col3, tab2 = st.columns(2)
                        
                
                with col3:
                    # Show secondary chart based on selection and button click
                    if generate_button:
                        st.write(f"### {sent_col} Distribution for {selected_topic}")
                        
                        # Filter data for the selected topic
                        filtered_df = df[df[topic_col] == selected_topic]
                        
                        if not filtered_df.empty:
                            # Create count by header for filtered data
                            header_counts = filtered_df[sent_col].value_counts().reset_index()
                            header_counts.columns = [sent_col, 'Count']
                            
                            fig2 = px.bar(
                                header_counts,
                                x=sent_col,
                                y='Count',
                                title=f"{sent_col} Distribution for {selected_topic}",
                                color=sent_col
                            )
                            
                            fig2.update_layout(height=500)
                            st.plotly_chart(fig2, use_container_width=True)
                            
                            # Show filtered data
    
                        else:
                            st.info(f"No data found for {selected_topic}")
                    else:
                        st.info("Select a topic and click 'Generate Secondary Chart' to view details")
                        
                # tab2 = st.columns(1)
                        
    
                if not filtered_df.empty:
                
                    with st.expander(f"Show summary for {selected_topic}"):
                        st.dataframe(filtered_df[['Summary', 'Subtopic','Sentiment', 'transcript']], use_container_width=True)
                
                
    with tab2 :
        with st.expander(f"Data Preview"):
            st.dataframe(df, use_container_width=True)
            
# else:
#     st.info("Please upload an Excel file to get started")

# Add footer with instructions
st.markdown("---")
st.write("""
### How to use this app:
1. Upload an Excel file and enter the api key
2. Click Analyse. It will take few mins or more based on the number of transcripts
3. It will share the topics discussed in the call
4. Click 'Generate Secondary Chart' to see the breakdown
""")
