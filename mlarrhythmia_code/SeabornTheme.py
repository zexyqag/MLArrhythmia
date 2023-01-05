import seaborn as sns

def setTheme():
    sns.set_theme(palette="pastel")
    sns.set(rc={
    'axes.facecolor':'#FFF8E1', 
    'figure.facecolor':'#465B65', 
    'ytick.color':'#FFF8E1', 
    'xtick.color':'#FFF8E1',
    'text.color':'#FFF8E1',
    'grid.color':'black',
    'axes.labelcolor':'#FFF8E1'
    })
