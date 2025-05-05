"""
                           DeepCache
               
DeepCache는 다음 BSD 3-Clause License에 따라 배포됩니다:

Copyright(c) 2019
                University of Minensota - Twin Cities
        Authors: Arvind Narayanan, Saurabh Verma, Eman Ramadan, Pariya Babaie, and Zhi-Li Zhang

모든 권리 보유.

소스 및 바이너리 형태로 재배포 및 사용은 다음 조건을 충족하는 경우에 허용됩니다:

1. 소스 코드의 재배포는 위의 저작권 고지, 이 조건 목록 및 다음 면책 조항을 유지해야 합니다.

2. 바이너리 형태로의 재배포는 위의 저작권 고지, 이 조건 목록 및 다음 면책 조항을 문서 및/또는 배포와 함께 제공되는 기타 자료에 포함해야 합니다.

3. 저작권 보유자의 이름이나 기여자의 이름은 사전 서면 허가 없이 이 소프트웨어에서 파생된 제품을 홍보하거나 보증하는 데 사용될 수 없습니다.

이 소프트웨어는 저작권 보유자 및 기여자에 의해 "있는 그대로" 제공되며, 상품성 및 특정 목적에 대한 적합성에 대한 묵시적 보증을 포함하되 이에 국한되지 않는 명시적 또는 묵시적 보증 없이 제공됩니다. 저작권 보유자 또는 기여자는 이 소프트웨어의 사용으로 인해 발생하는 직접적, 간접적, 부수적, 특별, 모범적 또는 결과적 손해(대체 상품 또는 서비스의 조달, 사용 손실, 데이터 또는 이익 손실, 또는 비즈니스 중단 포함)에 대해 책임을 지지 않습니다.

@author: Pariya Babaie (babai008@umn.edu) & Eman Ramadan (eman@cs.umn.edu)

설명:
    이 코드는 다음 논문을 기반으로 작성되었습니다:
    Tang, Wenting, et al. "MediSyn: A synthetic streaming media service workload generator."
    Proceedings of the 13th international workshop on Network and operating systems support
    for digital audio and video. ACM, 2003.
    https://dl.acm.org/citation.cfm?id=776327

    변수는 논문을 기반으로 하지만, 요청 분석 그래프에 맞게 요청을 생성하기 위해 일부 변경이 이루어졌습니다.
    이 데이터셋은 다음과 같은 특징을 통해 실제 트래픽을 시뮬레이션합니다:
        1- 새로운 객체가 다양한 시간에 도입됨
        2- 객체는 서로 다른 유형과 가변적인 수명을 가짐
        3- 각 객체에 대한 요청은 시간별 요청 비율에 따라 생성됨

입력:
    입력은 하드코딩되어 있으며 논문을 기반으로 초기화됩니다.
    {initialization} 섹션(Line 50)에서 변수를 수정할 수 있습니다.

흐름:
    1- 시간별 요청 비율을 로드하고, GENERATE_NEW_HOURLY_REQUEST_RATIO가 True인 경우 이를 재생성합니다.
    2- 객체 도입 날짜, 하루에 생성되는 객체 수, 하루 동안 도입된 객체 간의 간격을 생성합니다.
    3- 객체 빈도를 생성합니다.
    4- 객체와 그 속성(수명, 도입 시간, 종료 시간)을 생성합니다.
    5- 요청을 생성하고 내보냅니다.

출력:
    출력은 '../Datasets' 디렉토리에 다음 이름으로 생성됩니다:
     'mediSynDataset_x{'hourly_request_function_degree(예: 2)'}_O{'NUM_OF_OBJECTS'}.csv'
    요청 파일에는 요청될 객체 ID와 요청 시간이 포함됩니다.
"""

from __future__ import print_function
import pandas as pd
import numpy as np
import random
import math
import csv
import sys
import os


""""###########################  초기화  ####################################################################"""
NUM_OF_OBJECTS = 3500                             # 생성할 객체 수
lambdaFile = 'hourly_request_ratio.csv'           # 시간별 요청 비율 파일
lambdas = []                                      # 비동질적 포아송 분포를 위한 시간별 비율 저장
curTime = []                                      # 각 객체가 마지막으로 요청된 타임스탬프를 저장
objectPopularities = []                           # 객체의 인기도 저장
M = 178310                                        # HPC 데이터셋의 최대 빈도
traceType = 'HPC'
hourly_request_function_degree = 2               # 시간별 요청 비율을 설정하는 함수의 차수 (X^2)
dayGaps = []                                      # 일간 간격
numOfObjectsIntroduced = []                       # 하루에 생성된 객체 수
interArrivals = []                                # 하루 동안 도입된 객체 간의 간격
lifeSpanType = []                                 # 각 객체의 수명 유형 저장
ObjectsLifeSpan = []                              # 각 객체의 수명 길이 값 저장
requestGenInfo = {}                               # 각 객체의 요청 정보를 저장
startTimes = {}                                   # 객체 도입 시간을 기준으로 정렬
introductionOrder = []                            # 하루 동안 객체 도입 순서
sortedOnIntoTime = [] 
requests = []                                     # 생성된 요청
objectLengths = []
if sys.version_info[0] < 3:
    maxEndDay = -sys.maxint - 1
else:
    maxEndDay = -sys.maxsize - 1
WITH_INTRODUCTION = True                          # 객체가 다양한 시간에 도입되도록 허용하는 플래그
WITH_DAY_GAPS_INTRODUCTION = False               # True인 경우 객체 도입 날짜 간격을 추가
GENERATE_NEW_HOURLY_REQUEST_RATIO = False        # True인 경우 새로운 'hourly_request_ratio.csv' 파일 생성
MIN_REQ_PER_DAY_THRESHOLD = 1500                 # 하루에 각 객체에 대해 생성되는 최소 요청 수
MIN_OBJ_INTRODCUED_PER_DAY_THRESHOLD = 0.0035 * NUM_OF_OBJECTS  # 하루에 생성되는 최소 객체 수
MAX_OBJ_INTRODCUED_PER_DAY_THRESHOLD = 0.0095 * NUM_OF_OBJECTS  # 하루에 생성되는 최대 객체 수

# 출력 디렉토리 생성 (존재하지 않을 경우)
OUTPUTDIR = '../Datasets'
if not os.path.isdir(OUTPUTDIR):
    os.mkdir(OUTPUTDIR)

# hourly_request_ratio.csv 파일 존재 여부 확인
if not os.path.isfile('hourly_request_ratio.csv'):
    GENERATE_NEW_HOURLY_REQUEST_RATIO = True

# GENERATE_NEW_HOURLY_REQUEST_RATIO가 True인 경우 새로운 파일 생성
if GENERATE_NEW_HOURLY_REQUEST_RATIO:
    print('시간별 요청 비율 파일 생성 중...')
    rands = np.random.randint(1, 100, 24)
    rands = rands/float(np.sum(rands))
    index = np.arange(1, 25)

    res = 'hourly_request_ratio.csv'
    f = open(res, 'w+')
    for i in range(len(index)):
        if i != len(index)-1:
            f.write(str(index[i]) + ',' + str(rands[i])+'\n')
        else:
            f.write(str(index[i]) + ',' + str(rands[i]))
    f.close()


def initialize():
    """
    데이터셋 생성을 위한 초기화 함수.
    1. 시간별 요청 비율 로드
    2. 객체 도입 정보 생성
    3. 객체 인기도 생성
    4. 객체 생성
    5. 요청 생성
    """
    global curTime
    loadDiurnalRatios()
    print('데이터셋 생성을 위한 객체 생성 중...')
    generateObjectsIntroductionInfo(traceType)
    generatePopularities(traceType, int(NUM_OF_OBJECTS))
    generateObjects()

    print('데이터셋 생성을 위한 요청 생성 중...')
    curTime = [0] * NUM_OF_OBJECTS
    generateRequests()


"""################################ 시간별 비율 로드 #############################################################"""
def loadDiurnalRatios():
    """
    시간별 요청 비율을 로드하는 함수.
    'hourly_request_ratio.csv' 파일에서 데이터를 읽어와 lambdas 리스트에 저장.
    """
    with open(lambdaFile, "r+") as fi:
        for line in fi:
            tmpLambdas = float(line.rstrip('\n').rstrip('\r').split(',')[1])
            lambdas.append(tmpLambdas)
    fi.close()


"""###########################  객체 인기도 생성  ##################################################################"""
K = {'HPC': 30, 'HCL': 7}


def generatePopularities(traceType, N):
    """
    객체의 인기도를 생성하는 함수.
    Zipf 분포를 기반으로 객체의 인기도를 계산하여 objectPopularities 리스트에 저장.
    
    매개변수:
    - traceType: 트레이스 유형 ('HPC' 또는 'HCL')
    - N: 생성할 객체 수
    """
    zipalpha = 0.8
    k = K[traceType]
    for i in range(1, N+1):
        Mk = ((M-1)/k)+1
        tmp = (((float(Mk)/(math.pow((float(i+k-1)/k), zipalpha)))-1)*k)+1
        objectPopularities.append(tmp)


"""########################  객체 유형  ###########################################################################"""
def getObjectType():
    """
    객체 유형을 결정하는 함수.
    10%의 확률로 'news' 유형을 반환하고, 나머지 경우 'regular' 유형을 반환.
    """
    decision = random.uniform(0, 1)
    if decision <= 0.1:  # 10 %의 객체는 'news' 유형
        return 'news'
    else:
        return 'regular'


"""##################### 무작위 변량 생성 #################################################################"""
def generatePoissonVariate(rand, lambda_poisson):
    """
    포아송 분포를 따르는 무작위 변량을 생성하는 함수.
    """
    return -1 * (math.log(1-rand))/lambda_poisson


def generateParetoVariate(rand, alpha):
    """
    파레토 분포를 따르는 무작위 변량을 생성하는 함수.
    """
    return math.pow(1/rand, 1/alpha)


def generateParetoScaledVariate(rand, alpha, beta):
    """
    스케일된 파레토 분포를 따르는 무작위 변량을 생성하는 함수.
    """
    return beta / (math.pow((1 - rand), (1/alpha)))


def generateNormalVariate(mu, sigma):
    """
    정규 분포를 따르는 무작위 변량을 생성하는 함수.
    거부 방법을 사용하여 생성.
    """
    variateGenerated = False
    while not variateGenerated:
        u1 = random.uniform(0, 1)
        u2 = random.uniform(0, 1)
        x = -1*math.log(u1)
        if u2 > math.exp(-1*math.pow((x-1), 2)/2):
            continue
        else:
            u3 = random.uniform(0, 1)
            if u3 > 0.5:
                return mu+(sigma*x)
            else:
                return mu-(sigma*x)


def generateLogNormalVariate(mu, sigma):
    """
    로그 정규 분포를 따르는 무작위 변량을 생성하는 함수.
    거부 방법을 사용하여 생성.
    """
    variateGenerated = False
    while not variateGenerated:
        u1 = random.uniform(0, 1)
        u2 = random.uniform(0, 1)
        x = -1*math.log(u1)
        if u2 > math.exp(-1*math.pow((x-1), 2)/2):
            continue
        else:
            return math.exp(mu+(sigma*x))


def generateExponentialVariate(rand, a):
    """
    지수 분포를 따르는 무작위 변량을 생성하는 함수.
    """
    return -(1/a)*math.log(1-rand)


def generateRandVariate(dist, params, numOfVariates):
    """
    주어진 분포와 매개변수를 기반으로 무작위 변량을 생성하는 함수.
    
    매개변수:
    - dist: 분포 유형 ('pareto', 'paretoScaled', 'normal', 'logNormal', 'exp', 'poisson')
    - params: 분포 매개변수
    - numOfVariates: 생성할 변량 수
    
    반환값:
    - 생성된 무작위 변량 리스트
    """
    variates = []
    
    if dist is 'pareto':
        alpha = params['alpha']
        for i in range(numOfVariates):
            rand = random.uniform(0, 1)
            variates.append(generateParetoVariate(rand, alpha))

    if dist is 'paretoScaled':
        alpha = params['alpha']
        beta = params['beta']
        for i in range(numOfVariates):
            rand = random.uniform(0, 1)
            variates.append(generateParetoScaledVariate(rand, alpha, beta))

    elif dist is 'normal':
        mu = params['mu']
        sigma = params['sigma']
        for i in range(numOfVariates):
            variates.append(generateNormalVariate(mu, sigma))
            
    elif dist is 'logNormal':
        mu = params['mu']
        sigma = params['sigma']
        for i in range(numOfVariates):
            variates.append(generateLogNormalVariate(mu, sigma))
            
    elif dist is 'exp':
        mu = params['mu']
        for i in range(numOfVariates):
            rand = random.uniform(0, 1)
            variates.append(generateExponentialVariate(rand, mu))
    elif dist is 'poisson':
        mu = params['mu']
        for i in range(numOfVariates):
            rand = random.uniform(0, 1)
            variates.append(generatePoissonVariate(rand, mu))
    return variates


"""####################  객체 도입 정보 생성  ##################################################################"""
def generateObjectsIntroductionInfo(typeMode):
    """
    객체 도입 정보를 생성하는 함수.
    객체 도입 날짜 간격과 하루에 생성되는 객체 수를 생성.
    
    매개변수:
    - typeMode: 트레이스 유형 ('HPC' 또는 'HCL')
    """
    global NUM_OF_OBJECTS
    global numOfObjectsIntroduced

    tempNumOfObjectsIntroduced = []
    while sum(tempNumOfObjectsIntroduced) < NUM_OF_OBJECTS:
        if typeMode is 'HPC':
            if WITH_DAY_GAPS_INTRODUCTION:
                pareto_alpha_objectIntro_hpc = 1.0164
                object_intro_days_gap = generateRandVariate('pareto', {'alpha':pareto_alpha_objectIntro_hpc}, 1)[0]
                if object_intro_days_gap > 20:
                    object_intro_days_gap = 20
                dayGaps.append(object_intro_days_gap)
            else:
                dayGaps.append(1)
            
        else:
            exponential_mu_objectIntro_hpl = 4.2705
            object_intro_days_gap = generateRandVariate('exp', {'mu': exponential_mu_objectIntro_hpl}, 1)[0]
            dayGaps.append(object_intro_days_gap)
        
        # 하루에 생성되는 새로운 객체 수 (파레토 분포 기반)
        pareto_alpha_numOfObjectsGeneration = 0.8
        pareto_beta_numOfObjectsGeneration = MIN_OBJ_INTRODCUED_PER_DAY_THRESHOLD
        numOfObjects_intro_in_day = generateRandVariate('paretoScaled', {'alpha': pareto_alpha_numOfObjectsGeneration,
                                                        'beta': pareto_beta_numOfObjectsGeneration}, 1)[0]
        if numOfObjects_intro_in_day > MAX_OBJ_INTRODCUED_PER_DAY_THRESHOLD:
            numOfObjects_intro_in_day = MAX_OBJ_INTRODCUED_PER_DAY_THRESHOLD
        tempNumOfObjectsIntroduced.append(numOfObjects_intro_in_day)

    # 생성된 항목 정렬
    tempNumOfObjectsIntroduced.sort()
    extra_days = 0
    if len(tempNumOfObjectsIntroduced) % 7 != 0:
        extra_days = len(tempNumOfObjectsIntroduced) % 7
        for i in range(extra_days):
            # 이 객체들을 다른 도입 날짜에 추가하여 전체 주 데이터를 생성하기 위해 무작위 정수 생성
            added = False
            while not added:
                u = random.randint(extra_days+1, len(tempNumOfObjectsIntroduced) - 1)
                if tempNumOfObjectsIntroduced[i] + tempNumOfObjectsIntroduced[u] < MAX_OBJ_INTRODCUED_PER_DAY_THRESHOLD:
                    tempNumOfObjectsIntroduced[u] += tempNumOfObjectsIntroduced[i]
                    added = True

    # 다른 날짜에 추가된 후 추가 날짜 제외
    tempNumOfObjectsIntroduced = tempNumOfObjectsIntroduced[extra_days:]
    tempNumOfObjectsIntroduced.sort()

    # 정렬된 데이터를 다음과 같이 나누어 날짜를 채움
    # 이는 금요일에 더 많은 객체가 도입되고, 토요일, 일요일 순으로 도입됨을 의미.
    # 화요일에는 가장 적은 수의 객체가 도입됨.
    # 금 1, 토 2, 일 3, 목 4, 수 5, 월 6, 화 7
    weeks = int(len(tempNumOfObjectsIntroduced) / 7)
    FriIndex = weeks * 6
    SatIndex = weeks * 5
    SunIndex = weeks * 4
    MonIndex = weeks * 1
    TuesIndex = weeks * 0
    WedIndex = weeks * 2
    ThuIndex = weeks * 3

    for i in range(weeks):
        numOfObjectsIntroduced.append(tempNumOfObjectsIntroduced[MonIndex+i])
        numOfObjectsIntroduced.append(tempNumOfObjectsIntroduced[TuesIndex + i])
        numOfObjectsIntroduced.append(tempNumOfObjectsIntroduced[WedIndex + i])
        numOfObjectsIntroduced.append(tempNumOfObjectsIntroduced[ThuIndex + i])
        numOfObjectsIntroduced.append(tempNumOfObjectsIntroduced[FriIndex + i])
        numOfObjectsIntroduced.append(tempNumOfObjectsIntroduced[SatIndex + i])
        numOfObjectsIntroduced.append(tempNumOfObjectsIntroduced[SunIndex + i])

    # 하루 동안 객체 도입 간격 생성 (파레토 분포 기반)
    pareto_alpha_interArrival = 1.0073
    numOfDays = len(numOfObjectsIntroduced)
    for i in range(numOfDays):
        objectsCountInDay = int(np.round(numOfObjectsIntroduced)[i])
        if WITH_INTRODUCTION:
            interArrivals.append(generateRandVariate('pareto', {'alpha': pareto_alpha_interArrival}, objectsCountInDay))
        else:
            interArrivals.append([0]*objectsCountInDay)
    NUM_OF_OBJECTS = int(sum(np.round(numOfObjectsIntroduced)))


def generateObjectIntroductionOrder():
    """
    객체 도입 순서를 무작위로 생성하는 함수.
    """
    return np.random.permutation(range(len(objectPopularities)))+1


"""#########################  객체 수명 생성  ######################################################################"""
def generateLifeSpans(numOfObjects, objMode):
    """
    객체 수명을 생성하는 함수.
    로그 정규 분포와 파레토 분포를 기반으로 객체 수명을 계산하여 반환.
    
    매개변수:
    - numOfObjects: 생성할 객체 수
    - objMode: 객체 유형 리스트 ('regular' 또는 'news')
    
    반환값:
    - 객체 ID와 수명 튜플 리스트
    """
    logNormal_mu_mean = 3.0935
    logNormal_mu_std = 0.9612
    logNormal_sigma_mean = 1.1417
    logNormal_sigma_std = 0.3067
    pareto_alpha_mean = 1.7023
    pareto_alpha_std = 0.2092
    lifeSpans = []
   
    logNormalMu = generateRandVariate('normal', {'mu': logNormal_mu_mean, 'sigma': logNormal_mu_std}, 1)[0]
    logNormalSigma = generateRandVariate('normal', {'mu': logNormal_sigma_mean, 'sigma': logNormal_sigma_std}, 1)[0]
    
    paretoAlpha = generateRandVariate('normal', {'mu': pareto_alpha_mean, 'sigma': pareto_alpha_std}, 1)[0]
    
    for i in range(numOfObjects):
        if objMode[i] is 'regular':
            tmpLifeSpan = generateRandVariate('logNormal', {'mu': logNormalMu, 'sigma': logNormalSigma}, 1)[0]
        elif objMode[i] is 'news':
            tmpLifeSpan = generateRandVariate('pareto', {'alpha': paretoAlpha}, 1)[0]
        if tmpLifeSpan > 80:
            tmpLifeSpan = random.randint(2, 80)
        lifeSpans.append((i+1, tmpLifeSpan))
    return lifeSpans


"""#########################  객체 생성  ####################################################################"""
def normalizePopularities():
    """
    객체 인기도를 정규화하는 함수.
    """
    normalized = np.array(objectPopularities)/max(objectPopularities)
    return normalized


def getBinInterval(time):
    """
    주어진 시간에 대한 이진 간격을 계산하는 함수.
    """
    return (math.floor(time/float(3600)))/float(23)


def generateObjects():
    """
    객체를 생성하는 함수.
    객체 도입 순서, 수명, 도입 시간 등을 설정.
    """
    global ObjectsLifeSpan
    global introductionOrder
    global sortedOnIntoTime
    global maxEndDay
    normalizedPop = normalizePopularities()
    
    for i in range(len(normalizedPop)):
         lifeSpanType.append(getObjectType())
    # 튜플 (objID, LifeSpan), objID는 1부터 N까지
    ObjectsLifeSpan = generateLifeSpans(len(objectPopularities), lifeSpanType)
    introductionOrder = generateObjectIntroductionOrder()   # 객체 도입 순서 1부터 N까지
    for i in range(1, len(objectPopularities)+1):
        requestGenInfo[i] = {'startDay': 0, 'lifeSpan': 0, 'endDay': 0, 'arrivalTime': 0, 'type': '', 'freq': 0,
                             'unitPerDay': 0} # 1부터 N까지
        startTimes[i] = 0
        
    objCnt = 0  
    dayCnt = 0
    for i in range(len(numOfObjectsIntroduced)):
        dayTime = 0
        dayCnt = dayCnt+round(dayGaps[i])
        for j in range(int(np.round(numOfObjectsIntroduced)[i])):
            objIntroduced = introductionOrder[objCnt]
            dayTime = dayTime+interArrivals[i][j]
            requestGenInfo[objIntroduced]['startDay'] = dayCnt
            requestGenInfo[objIntroduced]['arrivalTime'] = dayTime 
            requestGenInfo[objIntroduced]['lifeSpan'] = ObjectsLifeSpan[objIntroduced-1][1]
            requestGenInfo[objIntroduced]['type'] = lifeSpanType[objIntroduced-1]
            requestGenInfo[objIntroduced]['freq'] = objectPopularities[objIntroduced-1]

            # 하루에 최소 요청 수를 생성하도록 설정
            if requestGenInfo[objIntroduced]['freq'] / requestGenInfo[objIntroduced]['lifeSpan'] \
                    < MIN_REQ_PER_DAY_THRESHOLD:
                # 업데이트할 숫자를 생성하기 위해 무작위 숫자 생성
                decision = random.uniform(0, 1)
                if decision <= 0.5:
                    # 객체 빈도 업데이트
                    life_span = random.randint(10, 80)
                    requestGenInfo[objIntroduced]['freq'] = life_span * MIN_REQ_PER_DAY_THRESHOLD
                    requestGenInfo[objIntroduced]['lifeSpan'] = life_span
                else:
                    # 객체 수명 업데이트
                    freq = random.randint(MIN_REQ_PER_DAY_THRESHOLD, 80*MIN_REQ_PER_DAY_THRESHOLD)
                    requestGenInfo[objIntroduced]['freq'] = freq
                    requestGenInfo[objIntroduced]['lifeSpan'] = freq / MIN_REQ_PER_DAY_THRESHOLD

            startTimes[objIntroduced] = dayCnt+getBinInterval(dayTime)

            requestGenInfo[objIntroduced]['endDay'] = requestGenInfo[objIntroduced]['lifeSpan'] + \
                                                      requestGenInfo[objIntroduced]['startDay']
            requestGenInfo[objIntroduced]['totalDens'] = math.pow(requestGenInfo[objIntroduced]['lifeSpan'],
                                                                  hourly_request_function_degree)

            objectLengths.append([objIntroduced, requestGenInfo[objIntroduced]['startDay'],
                                  requestGenInfo[objIntroduced]['lifeSpan'], requestGenInfo[objIntroduced]['endDay'],
                                  requestGenInfo[objIntroduced]['freq']])

            if requestGenInfo[objIntroduced]['endDay'] > maxEndDay:
                maxEndDay = requestGenInfo[objIntroduced]['endDay']
            objCnt = objCnt+1
    
    sortedOnIntoTime = sorted(startTimes, key=startTimes.get)

    
def generateDiurnalAccess(obj, diurnalRatio, dayCnt):
    """
    시간별 접근을 생성하는 함수.
    객체의 수명 동안 요청을 생성.
    
    매개변수:
    - obj: 객체 ID
    - diurnalRatio: 시간별 비율
    - dayCnt: 현재 날짜
    """
    global requests
    
    lifeTimeLeft = requestGenInfo[obj]['lifeSpan']

    if lifeTimeLeft > 1:
        lastDay = requestGenInfo[obj]['endDay']
        objCount = abs(requestGenInfo[obj]['freq']*(((math.pow(dayCnt-lastDay, hourly_request_function_degree)
                       - math.pow(lastDay-dayCnt+1, hourly_request_function_degree)))/requestGenInfo[obj]['totalDens']))
        requestGenInfo[obj]['lifeSpan'] = requestGenInfo[obj]['lifeSpan']-1
        for i in range(len(diurnalRatio)):
            tmpCount = int(np.round(objCount*diurnalRatio[i]))
            if tmpCount != 0:
                tmpLambda = (tmpCount/float(3600))
                reqInterArrivals = generateRandVariate('exp', {'mu': tmpLambda}, tmpCount)
                for tmpInter in reqInterArrivals:
                    requests.append((obj, (curTime[obj-1]+tmpInter)))
                    curTime[obj-1] = curTime[obj-1]+tmpInter
            
    else:
        lastDay = requestGenInfo[obj]['endDay']
        objCount = abs(requestGenInfo[obj]['freq']*(((math.pow(lastDay-dayCnt, hourly_request_function_degree)
                       - math.pow(lastDay-(dayCnt+requestGenInfo[obj]['lifeSpan']), hourly_request_function_degree))) /
                                                    requestGenInfo[obj]['totalDens']))
        spanToGenerate = int(math.floor(requestGenInfo[obj]['lifeSpan']*10))
        requestGenInfo[obj]['lifeSpan'] = 0
        
        for i in range(spanToGenerate):
            tmpCount = int(np.round(objCount*diurnalRatio[i]))
            if tmpCount != 0:
                tmpLambda = (tmpCount/float(3600))
            
                reqInterArrivals = generateRandVariate('exp', {'mu': tmpLambda}, tmpCount)
                for tmpInter in reqInterArrivals:
                    requests.append((obj, (curTime[obj-1]+tmpInter)))
                    curTime[obj-1] = curTime[obj-1]+tmpInter


"""#########################  요청 생성  ##################################################################"""
def generateRequests():
    """
    요청을 생성하는 함수.
    객체의 도입 시간과 수명에 따라 요청을 생성하고 파일에 저장.
    """
    global requests
    global curTime

    OUTPUTFILENAME = '{0}/mediSynDataset_x{1}_O{2}.csv'.format(OUTPUTDIR, hourly_request_function_degree, NUM_OF_OBJECTS)
    if not os.path.isfile(OUTPUTFILENAME):
        fi = open(OUTPUTFILENAME, 'w')
        fi.write('object_ID,request_time\n')
        fi.close()

    dayCount = requestGenInfo[sortedOnIntoTime[0]]['startDay']
    reqGendf = pd.DataFrame.from_dict(requestGenInfo, orient='index')
    reqGendf['objID'] = reqGendf.index

    while dayCount <= maxEndDay:
        objList = list(reqGendf[(reqGendf['startDay'] <= dayCount) & (reqGendf['endDay'] >= dayCount)]['objID'])
        for obj in objList:
            if curTime[obj-1] == 0:
                curTime[obj-1] = (dayCount*86400) + requestGenInfo[obj]['arrivalTime']

            generateDiurnalAccess(obj, lambdas, dayCount)
                
        dayCount = dayCount + 1
        if dayCount % 20 == 0:
            requests = sorted(requests, key=lambda x: x[1])
            saveRequestsToFile(OUTPUTFILENAME)
            requests = []
            print('{} Days Processed of {} Total Days'.format(dayCount, int(maxEndDay)))
    print('MediSyn Dataset Saved to Output file: {}'.format(OUTPUTFILENAME))


def saveRequestsToFile(OUTPUTFILENAME):
    """
    요청을 파일에 저장하는 함수.
    
    매개변수:
    - OUTPUTFILENAME: 출력 파일 이름
    """
    with open(OUTPUTFILENAME, 'a') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerows(requests)


"""##################################################################################################################"""


def main():
    """
    메인 함수.
    초기화 함수를 호출하여 데이터셋 생성을 시작.
    """
    initialize()


if __name__ == "__main__": main()
