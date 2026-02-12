import matplotlib.font_manager as fm

fm.fontManager.addfont("/home/vito/miniconda3/envs/gnff_env/fonts/times.ttf")

for f in fm.fontManager.ttflist:
    if "Times" in f.name:
        print(f.name)