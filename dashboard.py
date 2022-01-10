from os import write
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode()


############################## LOAD DATA ##############################

#df = pd.read_csv('Data/Output/final_data.csv')
df = pd.read_csv('final_data.csv')


################### HELPER FUNCTIONS TO PROCESS DATA ##################

def get_breaks(content, length):
    data = ""
    words = content.split(' ')
    total_chars = 0
    # add break every length characters
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            data = data + "<br>" + words[i]
            total_chars = 0
        else:
            data = data + " " + words[i]
    return data



def format_abstract(abstract):
    
    if len(abstract.split()) == 0: 
        return "Abstract not provided."

    elif len(abstract.split(' ')) > 150:
        info = abstract.split(' ')[:150]
        return ' '.join(info+['...'])

    else:
        return abstract

############################# PROCESS DATA #############################

df['metadata'] = df[['title', 'journal', 'abstract']].apply(lambda x: x[0]+' '+x[1]+' '+x[2],axis=1)
df['abstractFormatted'] = df['abstract'].apply(format_abstract)

df['abstract_formatted'] = df['abstractFormatted'].apply(lambda x: get_breaks(x,35))
df['title_'] = df['title'].apply(lambda x: get_breaks(x,35))
df['authors_'] = df['authors'].apply(lambda x: get_breaks(x,35))
df = df.drop(['title', 'authors', 'abstract', 'abstractFormatted'], axis=1)




################## FUNCTION TO PLOT GIVEN CLUSTER ####################

def plot(df, fig, cluster, keyword=None):
        
    data = df[df['num_cluster']==cluster]
    
    if keyword==None: 
        fig.add_scatter(x = data['tSNE1'], y = data['tSNE2'], name=f'CL-{cluster}', mode='markers', 
                        customdata=data[['title_', 'abstract_formatted', 'authors_', 'journal', 'doi']])
    else: 
        data = filtered_keyword(df, cluster, keyword)
        fig.add_scatter(x = data['tSNE1'], y = data['tSNE2'], name=f'CL-{cluster}, Keyword: {keyword}',
                        mode='markers', 
                        customdata=data[['title_', 'abstract_formatted', 'authors_', 'journal', 'doi']])



    fig.update_traces(
        marker=dict(size=8,
                    line=dict(width=.8,color='black')),
        
        hovertemplate = (
                                        '<i><b>Title</i></b>: <b>%{customdata[0]}</b><br>'+
                                        '<br>Abstract: %{customdata[1]}<br>'+
                                        '<br>Authors: %{customdata[2]}'+
                                        '<br>Journal: %{customdata[3]}'+
                                        '<br>Link: %{customdata[4]}'))



    fig.update_layout(autosize=False, width=1200, height=700, 
                    title= {
                            'text': 'Clustering of Covid Literature using tSNE and KMeans.',
                            'y':0.9,
                            'x':0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'},
                    hoverlabel=dict(bgcolor="white", font_size=12, font_family="TimesNewRoman"))
    return fig





def get_keywords(df, cluster):
    data = df[df['num_cluster']==cluster]
    keywords = data['keywords'].unique()
    return ['All'] + eval(keywords[0])
    # print('Top Keywords are : ', eval(keywords[0]))


    

    
def filtered_keyword(df, cluster, keyword):
    
    data = df[df['num_cluster']==cluster]
    
    indexes = []
    for text in data['metadata']:
        words = text.split()
        for word in words:
            if word == keyword :
                df_text = data[data['metadata'] == text]
                indexes.append((df_text.index)[0])
                continue
    
    return df.iloc[indexes]


def default_plot():
    fig = go.Figure()
    for i in range(14):
        fig = plot(df, fig, i)
    st.plotly_chart(fig)

###################### FUNCTIONS DEFINED #############################



################################# INTRO #################################
st.set_page_config(page_title="Covid-19 Literature Clustering")


# define title
st.markdown("<h1 style='text-align: left; color: white;'>Covid-19 Literature Clustering</h1>", unsafe_allow_html=True)

desc_= '''
Related research papers have been grouped together to make it easier for health professionals to access 
relevant research. Clustering can be used to create a tool that finds similar articles based on a target 
article. It can also cut down on the number of articles you have to read by allowing you to concentrate 
on a small set of articles rather than a large number of different sorts. We explain how clustering can 
be used to do this in this graph.
'''
# st.markdown("<h1 style='text-align: center; color: #949494;'>Covid-19 Literature Clustering</h1>", unsafe_allow_html=True)
st.markdown("<i style='text-align: left; color: white;'>" +desc_+ "</i>", unsafe_allow_html=True)

# define slider to define the cluster number
st.sidebar.write('The target cluster can be filtered using the slider below. Simply move the slider to the desired cluster number to see the plots in that cluster. To see everything, slide back to 0.')
c = st.sidebar.slider("Choose the cluster", 0,14,0)

def visualize(c): 

    if c > 0:
        st.sidebar.write('Filter the paper by selecting the relevant keyword. It will search the title, journal and abstract for the given keyword. Select "All" to reset the plot.')
        word = st.sidebar.selectbox('Filter by keyword', get_keywords(df, c-1))

        if word == 'All':
            fig = go.Figure()
            fig = plot(df, fig, c-1)
            st.plotly_chart(fig)
            st.sidebar.write('Top 10 keywords are :', get_keywords(df, c-1)[1:11])
        elif word!='All':
            fig = go.Figure()
            plot(df, fig, c-1, keyword=word)
            st.plotly_chart(fig, use_container_width=True)
        

    else:
        default_plot()

visualize(c)
