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

@author: Eman Ramadan (eman@cs.umn.edu) & Pariya Babaie (babai008@umn.edu)

설명:
    이 코드는 여러 세션에서 객체의 인기와 요청 간격 분포가 변경되는 합성 데이터를 생성합니다.

입력:
    - alphaSet: 각 세션에 대한 객체 인기 Zipf 분포 값.
    - NUM_OF_OBJECTS: 세션당 요청 생성에 사용되는 객체 수, Object_ID는 1부터 NUM_OF_OBJECTS까지.
    - NUM_OF_REQUESTS_PER_SESSION: 세션당 요청 수.
    총 요청 수는 num_of_sessions * NUM_OF_REQUESTS_PER_SESSION과 대략 동일합니다.
    이러한 입력은 하드코딩되어 있으므로 스크립트에 인수를 전달할 필요가 없습니다. 직접 실행할 수 있습니다.
    {initialization} 섹션(Line 45)에서 이러한 매개변수를 수정할 수 있습니다.

흐름:
    1- alpha 세트의 각 값에 대해 'NUM_OF_REQUESTS_PER_SESSION' 요청이 각 세션에서 생성됩니다.
    2- 객체의 순위(따라서 인기도)가 각 세션에서 순열화됩니다. 즉, 첫 번째 세션에서 인기가 있는 객체는 두 번째 세션에서 다른 인기를 가질 수 있습니다. 간격 분포도 세션마다 다릅니다.
    3- 각 세션에서 각 객체에 대한 요청 세트가 생성되고 이전에 생성된 요청에 추가됩니다.
    4- 모든 요청은 요청 시간에 따라 정렬된 객체 ID와 요청 시간을 포함하는 데이터 프레임에 저장됩니다.

출력:
    출력은 '../Datasets' 디렉토리에 다음 이름으로 생성됩니다:
     'syntheticDataset_O{NUM_OF_OBJECTS}.csv'
    요청 파일에는 요청될 객체 ID와 요청 시간이 포함됩니다.
"""

from __future__ import print_function
import random
import math
import os
import numpy as np
import pandas as pd


""""###########################  초기화  ####################################################################"""
alphaSet = [0.8, 1, 0.5, 0.7, 1.2, 0.6]  # 각 세션에 대한 Zipf 분포의 alpha 값
NUM_OF_OBJECTS = 50  # 요청 생성에 사용되는 객체 수
NUM_OF_REQUESTS_PER_SESSION = 50000  # 세션당 요청 수

OUTPUTDIR = '../Datasets'  # 출력 파일이 저장될 디렉토리
OUTPUTFILENAME = '{}/syntheticDataset_O{}.csv'.format(OUTPUTDIR, NUM_OF_OBJECTS)  # 출력 파일 이름
if not os.path.isdir(OUTPUTDIR):
    os.mkdir(OUTPUTDIR)


def generate_requests():
    """
    요청 데이터를 생성하는 함수.
    각 세션에 대해 객체의 순위와 요청 간격 분포를 변경하며 요청 데이터를 생성합니다.
    
    주요 작업:
    1. 각 세션에 대해 Zipf 분포의 alpha 값을 기반으로 요청 데이터를 생성.
    2. 세션 간 객체의 순위를 무작위로 변경.
    3. 이전 세션의 마지막 요청 시간 이후부터 요청 시간을 조정.
    4. 모든 요청 데이터를 하나의 데이터프레임에 병합하고 정렬.
    5. 결과를 CSV 파일로 저장.
    """
    print('데이터셋 요청 생성 중...')
    for i in range(len(alphaSet)):
        print('세션 번호 {} / {} 처리 중'.format(i + 1, len(alphaSet)))
        alpha = alphaSet[i]

        # 세션별로 분포를 무작위로 선택 (Poisson 또는 Pareto)
        if random.randint(1, 2) % 2 == 0:
            dist = 'P'    # Poisson 분포
        else:
            dist = 'R'    # Pareto 분포

        if i != 0:
            # 새로운 순위 순열을 사용하여 객체 순위를 변경
            perm = np.random.permutation(NUM_OF_OBJECTS) + 1

        print('Alpha = {}, 분포 = {}'.format(alpha, 'Poisson' if dist == 'P' else 'Pareto'))
        requests = generate_session_requests(alpha, dist)  # 요청 데이터 생성 (객체 ID, 요청 시간 반환)

        df = pd.DataFrame(requests, columns=['requests'])
        # 요청 데이터를 두 개의 열(object_ID, request_time)로 분리
        parts = df["requests"].str.split(",", expand=True)

        # 객체 ID 열 생성
        df["object_ID"] = parts[0]
        df["object_ID"] = df["object_ID"].astype(int)

        # 요청 시간 열 생성
        df["request_time"] = parts[1]
        df["request_time"] = df["request_time"].astype(float)

        if i == 0:
            requestsdf = df[['object_ID', 'request_time']].copy(deep=True)
        else:
            # 현재 세션에서 각 객체의 순위를 순열화
            for cur_obj_id in df.groupby('object_ID').groups.keys():  # cur_obj_id: 1부터 NUM_OF_OBJECTS까지
                new_obj_id = perm[cur_obj_id - 1]
                # 현재 객체 ID를 새로운 객체 ID로 대체
                new_obj_id_col = np.full((len(df.groupby('object_ID').groups[cur_obj_id])), new_obj_id)

                # 이전 세션의 마지막 요청 시간부터 시작하도록 요청 시간 조정
                last_req_time = lastReqs.loc[lastReqs['object_ID'] == cur_obj_id]['request_time_max'].tolist()[0]
                new_request_times_col = df.loc[df['object_ID'] == cur_obj_id, ['request_time']] + last_req_time
                new_request_times_col = new_request_times_col['request_time'].tolist()

                # 이 객체에 대한 임시 데이터 프레임 생성
                tmp_df = pd.DataFrame({'object_ID': new_obj_id_col, 'request_time': new_request_times_col})

                # 기존 요청 데이터와 병합
                requestsdf = requestsdf.append(tmp_df, ignore_index=True)

        # 다음 세션을 위해 각 객체 ID의 마지막 요청 시간을 가져옴
        lastReqs = requestsdf.groupby(['object_ID']).agg({'request_time': ['max']})
        # 그룹화 후 열 이름 평탄화
        lastReqs.columns = ['_'.join(col).strip() for col in lastReqs.columns.values]
        lastReqs['object_ID'] = lastReqs.index
        # 이제 lastReqs 데이터 프레임에는 각 객체 ID와 마지막 요청 시간이 포함됨

    # 모든 요청 데이터를 요청 시간에 따라 정렬
    requestsdf.sort_values('request_time', ascending=True, inplace=True)

    # 결과를 CSV 파일로 저장
    f = open(OUTPUTFILENAME, 'w')
    requestsdf.to_csv(f, header=True, index=False)
    f.close()
    print('합성 데이터셋이 출력 파일에 저장되었습니다: {}'.format(OUTPUTFILENAME))
    del requestsdf


"""###########################  객체 인기 생성  ##################################################################"""
def generate_object_popularity_zipf(zipalpha):
    """
    Zipf 분포를 사용하여 객체의 인기도를 생성하는 함수.
    
    매개변수:
    - zipalpha: Zipf 분포의 alpha 값.
    
    주요 작업:
    1. Zipf 분포를 기반으로 각 객체의 인기도를 계산.
    2. 객체 ID와 해당 인기도를 반환.
    """
    N = NUM_OF_OBJECTS
    denom = 0.0
    for i in range(N):
        denom += 1.0/pow((i+1), zipalpha)
    objects_zipf_pdf = []
    for i in range(N):
        item = 1.0/pow((i+1), zipalpha)
        item /= denom
        con_index = i+1
        objects_zipf_pdf.append((con_index, item))
    objects_zipf_pdf = sorted(objects_zipf_pdf, key=lambda a: a[0])
    return objects_zipf_pdf


"""###########################  세션 요청 생성  ###################################################################"""
def generate_session_requests(zipalpha, distr):
    """
    세션 요청 데이터를 생성하는 함수.
    
    매개변수:
    - zipalpha: Zipf 분포의 alpha 값.
    - distr: 요청 간격 분포 (Poisson 또는 Pareto).
    
    주요 작업:
    1. Zipf 분포를 기반으로 객체의 인기도를 생성.
    2. 각 객체에 대해 요청 간격을 생성 (Poisson 또는 Pareto 분포 사용).
    3. 요청 데이터를 시간 순서대로 정렬하여 반환.
    """
    requests = []

    # 객체 인기도 생성
    objects_zipf_pdf = generate_object_popularity_zipf(zipalpha)

    # 최소 인기 객체에 대한 요청 수를 기반으로 최대 시뮬레이션 시간 계산
    simulation_time_end = 0
    N = NUM_OF_OBJECTS - 1
    reqs_N = int(math.ceil(objects_zipf_pdf[N][1] * NUM_OF_REQUESTS_PER_SESSION))
    ctr = 0
    cur_t = 0
    while ctr < reqs_N:
        rand = np.random.uniform()
        if distr == 'P':
            t = generate_poisson_distribution_from_CDF(rand, objects_zipf_pdf[N][1])
        elif distr == 'R':
            t = generate_pareto_distribution_from_CDF(rand, objects_zipf_pdf[N][1])
        ctr += 1
        cur_t += t
        simulation_time_end = cur_t
        req_str = str(N+1) + ',' + str(cur_t)
        requests.append(req_str)

    # 나머지 객체에 대한 요청 생성
    for i in range(NUM_OF_OBJECTS - 1):
        cur_t = 0
        while cur_t < simulation_time_end:
            rand = np.random.uniform()
            if distr == 'P':
                t = generate_poisson_distribution_from_CDF(rand, objects_zipf_pdf[i][1])
            elif distr == 'R':
                t = generate_pareto_distribution_from_CDF(rand, objects_zipf_pdf[i][1])
            cur_t += t
            req_str = str(i+1) + ',' + str(cur_t)
            requests.append(req_str)

    requests = sorted(requests, key=lambda a: float(a.split(',')[1]))
    return requests


def generate_poisson_distribution_from_CDF(rand, lambda_poisson):
    """
    Poisson 분포를 사용하여 요청 간격을 생성하는 함수.
    
    매개변수:
    - rand: 0과 1 사이의 난수.
    - lambda_poisson: Poisson 분포의 lambda 값.
    
    주요 작업:
    1. Poisson 분포의 CDF를 역으로 계산하여 요청 간격을 생성.
    """
    return -1 * (math.log(1-rand))/lambda_poisson


def generate_pareto_distribution_from_CDF(rand, lambda_pareto):
    """
    Pareto 분포를 사용하여 요청 간격을 생성하는 함수.
    
    매개변수:
    - rand: 0과 1 사이의 난수.
    - lambda_pareto: Pareto 분포의 lambda 값.
    
    주요 작업:
    1. Pareto 분포의 CDF를 역으로 계산하여 요청 간격을 생성.
    """
    # Pareto 분포의 beta 값은 2로 설정
    return (1/math.sqrt(1-rand) - 1)/lambda_pareto


"""##################################################################################################################"""


def main():
    """
    스크립트의 진입점 함수.
    요청 데이터를 생성하는 generate_requests() 함수를 호출합니다.
    """
    generate_requests()

if __name__ == "__main__": main()
