'''
1) (time_window + 2) x 11(피쳐 수) 로 Reshape한 다음에 np.array로 만들어서 학습
2) 학습할 수 있는 형태로 만들고 모델에 바로 넣기
'''
path = './Aion_april_w4_bot .csv_processed.csv'
temp = []

#임시로 사용할 index 생성
for i in range(0,11*32):
    temp.append(str(i))

data = pd.read_csv(path, names=temp)
data = pd.DataFrame(data)

bot_dataset = []

for j in range(len(data)):
    indi_data = data.iloc[j].tolist()
    np_data = np.array([np.array(indi_data).astype(np.float32).reshape(32,11)])
    
    #전처리 뻑나지않게 바꾸면, 밑에 부분은 굳이 안 추가해도 됨
    np_data = np_data[0] #dimension 하나 낮춰주고
    np_data = np.delete(np_data,(-1), axis=0)
    np_data = np.delete(np_data,(-1), axis=0)
    
    ### 여기서 통계값 넣어주고
    
    #일반 List에 넣어줌
    bot_dataset.append(np_data)