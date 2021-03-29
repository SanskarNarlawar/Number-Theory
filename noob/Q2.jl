using CSV
using LinearAlgebra
using Statistics
using DataFrames
using Random
dataset = CSV.read("data/FlightDelays.csv")

crs_dep_time = dataset.CRS_DEP_TIME
crs_dep_min = 60*floor.(crs_dep_time/100) + (crs_dep_time - 100*floor.(crs_dep_time/100))
crs_dep_sin = sin.((2*pi/1440)*crs_dep_min)
crs_dep_cos = cos.((2*pi/1440)*crs_dep_min)

carrier = dataset.CARRIER
CO = zeros(length(carrier))
DH = zeros(length(carrier))
DL = zeros(length(carrier))
MQ = zeros(length(carrier))
OH = zeros(length(carrier))
RU = zeros(length(carrier))
UA = zeros(length(carrier))
US = zeros(length(carrier))
for i in 1:length(carrier)
    if carrier[i]=="CO"
        CO[i]=1
    elseif carrier[i]=="DH"
        DH[i]=1
    elseif carrier[i]=="DL"
        DL[i]=1
    elseif carrier[i]=="MQ"
        MQ[i]=1
    elseif carrier[i]=="OH"
        OH[i]=1
    elseif carrier[i]=="RU"
        RU[i]=1
    elseif carrier[i]=="UA"
        UA[i]=1    
    elseif carrier[i]=="US"
        US[i]=1
    end
end

dep_time = dataset.DEP_TIME
dep_min = 60*floor.(dep_time/100) + (dep_time - 100*floor.(dep_time/100))
dep_sin = sin.((2*pi/1440)*dep_min)
dep_cos = cos.((2*pi/1440)*dep_min)

dest=dataset.DEST
JFK=zeros(length(dest))
LGA=zeros(length(dest))
EWR=zeros(length(dest))
for i in 1:length(dest)
    if dest[i]=="JFK"
        JFK[i]=1
    elseif dest[i]=="LGA"
        LGA[i]=1
    elseif dest[i]=="EWR"
        EWR[i]=1
    end
end

distance=dataset.DISTANCE
Dist=unique(distance)
dist=zeros(length(distance),length(Dist))
for i in 1:length(distance)
    for j in 1:length(Dist)
        if distance[i]==Dist[j]
            dist[i,j]=1
        end
    end
end

flight_num=dataset.FL_NUM
flnum=unique(flight_num)
fl_num=zeros(length(flight_num),length(flnum))
for i in 1:length(flight_num)
    for j in 1: length(flnum)
        if flight_num[i]==flnum[j]
            fl_num[i,j]=1
        end
    end
end

origin=dataset.ORIGIN
DCA=zeros(length(origin))
IAD=zeros(length(origin))
BWI=zeros(length(origin))
for i in 1:length(origin)
    if origin[i]=="DCA"
        DCA[i]=1
    elseif origin[i]=="IAD"
        IAD[i]=1
    elseif origin[i]=="BWI"
        BWI[i]=1
    end
end

weather=dataset.Weather

day_week = dataset.DAY_WEEK
monday = zeros(length(day_week))
tuesday = zeros(length(day_week))
wednesday = zeros(length(day_week))
thursday = zeros(length(day_week))
friday = zeros(length(day_week))
saturday = zeros(length(day_week))
sunday = zeros(length(day_week))
for i in 1:length(day_week)
    if day_week[i]==1
        monday[i]=1
    elseif day_week[i]==2
        tuesday[i]=1
    elseif day_week[i]==3
        wednesday[i]=1
    elseif day_week[i]==4
        thursday[i]=1
    elseif day_week[i]==5
        friday[i]=1
    elseif day_week[i]==6
        saturday[i]=1
    elseif day_week[i]==7
        sunday[i]=1    
    end
end

day_month=dataset.DAY_OF_MONTH

tail_num=dataset.TAIL_NUM
tlnum=unique(tail_num)
tl_num=zeros(length(tail_num),length(tlnum))
for i in 1:length(tail_num)
    for j in 1: length(tlnum)
        if tail_num[i]==tlnum[j]
            tl_num[i,j]=1
        end
    end
end

categorical_flight_status=dataset.Flight_Status
flight_status=zeros(length(categorical_flight_status))
for i in 1:length(categorical_flight_status)
    if categorical_flight_status[i]=="ontime"
        flight_status[i]=0
    elseif categorical_flight_status[i]=="delayed"
        flight_status[i]=1
    end
end

x0=ones(length(flight_status))
X = cat(x0,crs_dep_sin,crs_dep_cos,CO,DH,DL,MQ,OH,RU,UA,US,dep_sin,dep_cos,JFK,LGA,EWR,dist,fl_num,DCA,IAD,BWI,weather,monday,tuesday,wednesday,thursday,friday,saturday,sunday,day_month,tl_num,dims=2)
Y = flight_status
data=cat(X,Y,dims=2)

function partitionTrainTest(data, at = 0.7)
    n = size(data,1)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx,:], data[test_idx,:]
end

Random.seed!(101)
data_train,data_test = partitionTrainTest(data,0.6)
Xtrain = data_train[:,1:size(X,2)]
Xtest = data_test[:,1:size(X,2)]
Ytrain = data_train[:,size(data,2)]
Ytest = data_test[:,size(data,2)]

function normalise(X)
    X=(X - mean(X)*ones(length(X)))/(((sum((X - mean(X)*ones(length(X))).^2))/length(X))^0.5)
    return X
end

Xtrain[:,size(X,2)-size(tl_num,2)]=normalise(Xtrain[:,size(X,2)-size(tl_num,2)])
Xtest[:,size(X,2)-size(tl_num,2)]=normalise(Xtest[:,size(X,2)-size(tl_num,2)])

function sigmoid(z)
    g = zeros(size(z))
    for i in 1:size(z,1)
        for j in 1:size(z,2)
            g[i,j]= 1/(1+exp(-z[i,j]));
        end
    end
    return g
end

function costFunction(X,Y,theta)
    m = length(Y)
    I = ones(m,1)
    J = (-1/m)*sum((Y.*log(sigmoid(X*theta)))+(I-Y).*log(I-sigmoid(X*theta)))
    return J
end

function gradientDescent(X,Y,theta,alpha,iterations)
    m=length(Y)
    for iteration in 1:iterations
        grad = (1/m)*X'*(sigmoid(X*theta)-Y)
        theta = theta - alpha*grad
    end
    return theta
end

theta=zeros(size(X,2))
alpha=1
newTheta=gradientDescent(Xtrain,Ytrain,theta,alpha,5000)
YPred=sigmoid(Xtest*newTheta)
YPred_test=zeros(length(Ytest))
threshold=0.6
for i in 1:length(Ytest)
    if YPred[i]>threshold
        YPred_test[i]=1
    end
end
function acc(Y_Test,Y_Pred)
    count=0
    for i in 1:length(Y_Test)
        if Y_Test[i]==Y_Pred[i]
        count=count+1
        end
    end
    accu = (count/length(Ytest))*100
    return accu
end
accuracy=acc(Ytest,YPred_test)

println("Accuracy on Test Set = ", accuracy)

function ac(Y1,Y2)
    count1=0
    for i in 1:length(Y1)
        if Y1[i]==Y2[i]
        count1=count1+1
        end
    end
    ac1 = (count1/length(Y1))*100
    return ac1
end
YPred_train=sigmoid(Xtrain*newTheta)
YPred_train_binary=zeros(length(Ytrain))
threshold=0.6
for i in 1:length(Ytrain)
    if YPred_train[i]>threshold
        YPred_train_binary[i]=1
    end
end
accuracy_train = ac(Ytrain,YPred_train_binary)

println("Accuracy on Training Set = ", accuracy_train)


