import pandas as pd
import os
def save_file(df):
    save_path='../../data/modified/'
    df.to_csv(os.path.join(save_path+'data_to_work_on.csv'),index=False)
    return os.path.join(save_path+'data_to_work_on.csv')


if __name__=='__main__':
    #import the raw data
    df=pd.read_csv('../../data/raw/articles1.csv')
    # Drop the unwanted columns
    df.drop(columns=['Unnamed: 0','id','publication','author','date','year','month','url'],inplace=True)
    # Renaming and Reordering the columns
    df.rename(columns={'title':'Headline','content':'Article'},inplace=True)
    df=df.reindex(columns=['Article','Headline'])
    
    '''
    The headlines includes the publication name in them, which we don't want so we'll remove them
    e.g.- House Republicans Fret About Winning Their Health Care Suit - The New York Times 
    '''
    df['Headline']=df['Headline'].apply(lambda x:x.split(' - ')[0].strip())

    #save the file
    path=save_file(df)
    print(f'Data prepared for further steps and and saved to {path}')
    