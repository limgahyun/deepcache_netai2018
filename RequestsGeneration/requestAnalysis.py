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

이 소프트웨어는 저작권 보유자 및 기여자에 의해 "있는 그대로" 제공되며, 상품성 및 특정 목적에 대한 적합성에 대한 묵시적 또는 묵시적 보증 없이 제공됩니다. 저작권 보유자 또는 기여자는 이 소프트웨어의 사용으로 인해 발생하는 직접적, 간접적, 부수적, 특별, 모범적 또는 결과적 손해(대체 상품 또는 서비스의 조달, 사용 손실, 데이터 또는 이익 손실, 또는 비즈니스 중단 포함)에 대해 책임을 지지 않습니다.


@author: Eman Ramadan (eman@cs.umn.edu)


설명:
    이 코드는 생성된 요청 데이터를 분석하여 각 객체의 속성(도입 시점, 빈도, 수명 등)을 추출합니다.
    또한 매 시간 요청된 고유 활성 객체 수를 추출합니다.


전처리 스크립트:
    requestAnalysis.py 스크립트를 실행하기 전에 다음 스크립트 중 하나를 실행해야 합니다:
    1. generateSyntheticDataset.py: 합성 데이터셋 생성
    2. generateMediSynDataset.py: MediSyn 논문에 따라 합성 데이터셋 생성


입력:
    입력 디렉토리는 '../Datasets'입니다:
    1- REQUESTFILENAME: 분석할 요청 파일 (Line 41에서 설정)
    2- FORCE_GENERATE_BINS: bin 파일 재생성을 강제하는 플래그, 기본값은 False
    3- FORCE_GENERATE_PROPERTIES: 객체 속성 파일 재생성을 강제하는 플래그, 기본값은 False


출력:
    출력 파일은 '../Datasets' 디렉토리에 생성됩니다:
    1- {RequestFile}_bins.csv: 각 시간대의 고유 객체 수를 나타냅니다.
                            형식: {binID, uniqueObjNum, binMinRequestTime, binMaxRequestTime}
    2- {RequestFile}_properties.csv: 각 객체의 속성을 포함합니다.
                            형식: {object_ID, frequency, lifeSpan, minRequestTime, maxRequestTime, start_day, end_day}
"""

from __future__ import print_function
import pandas as pd
import numpy as np
import os

# bin 파일 및 속성 파일 재생성 여부를 설정하는 플래그
FORCE_GENERATE_BINS = False  # bin 파일 재생성 강제 여부
FORCE_GENERATE_PROPERTIES = False  # 객체 속성 파일 재생성 강제 여부

# 요청 파일 및 출력 파일 경로 설정
REQDIR = '../Datasets/'  # 요청 파일이 위치한 디렉토리
REQUESTFILENAME = 'mediSynDataset_x2_O3488.csv'  # 분석할 요청 파일 이름
REQUESTPATH = REQDIR + REQUESTFILENAME  # 요청 파일 전체 경로
BINFILENAME = REQDIR + REQUESTFILENAME[:-4] + '_bins.csv'  # bin 파일 경로
PROPERTIES_FILENAME = REQDIR + REQUESTFILENAME[:-4] + '_properties.csv'  # 속성 파일 경로
BIN_SECONDS_WIDTH = 3600  # bin의 시간 간격 (초 단위, 1시간)

# 요청 파일 로드
print('요청 파일 로드 중...')
reqdf = pd.read_csv(REQUESTPATH, sep=',')  # 요청 파일 읽기
print('요청 파일을 시간 순으로 정렬 중...')
reqdf.sort_values(by=['request_time'], inplace=True)  # 요청 파일을 요청 시간 기준으로 정렬
print('요청 파일 정렬 완료')

# 1시간 간격의 bin 생성
if not os.path.isfile(BINFILENAME) or FORCE_GENERATE_BINS:
    bins = np.arange(np.ceil(reqdf.request_time.min()), np.ceil(reqdf.request_time.max()), BIN_SECONDS_WIDTH)
    print('binning 프로세스 시작...')
    reqdf['binID'] = pd.cut(reqdf['request_time'], bins, labels=np.arange(0, len(bins)-1))  # 요청 시간을 bin에 매핑

    # 각 bin에 대해 고유 객체 수 및 요청 시간 범위 계산
    grp = reqdf.groupby(['binID']).agg({'object_ID': {'uniqueObjNum': lambda x: x.nunique()},
                                        'request_time': ['min', 'max']})
    grp.reset_index(level=0, inplace=True)

    # 열 이름 정리
    cols = list()
    for k in grp.columns:
        if k[1] == '':
            cols.append(k[0])
        else:
            cols.append(k[1])
    grp.columns = cols

    # NaN 값 제거 및 데이터 정리
    filtered = grp.dropna()
    filtered["uniqueObjNum"] = filtered["uniqueObjNum"].apply(int)  # 고유 객체 수를 정수형으로 변환
    filtered.rename(columns={'min': 'binMinRequestTime'}, inplace=True)  # 최소 요청 시간 열 이름 변경
    filtered.rename(columns={'max': 'binMaxRequestTime'}, inplace=True)  # 최대 요청 시간 열 이름 변경
    filtered.to_csv(BINFILENAME, index=False)  # bin 파일 저장
    del filtered

# 객체 속성 계산
if not os.path.isfile(PROPERTIES_FILENAME) or FORCE_GENERATE_PROPERTIES:
    # 객체 빈도 계산
    print('객체 빈도 계산 중...')
    objfreqdf = (reqdf['object_ID'].value_counts()).to_frame()  # 객체 ID별 요청 수 계산
    objfreqdf.rename(columns={'object_ID': 'frequency'}, inplace=True)  # 열 이름 변경
    objfreqdf['object_ID'] = objfreqdf.index  # 객체 ID를 인덱스에서 열로 이동

    # 객체 수명 및 도입 날짜 계산
    print('객체 수명 및 도입 날짜 계산 중...')
    reqdf.sort_values(by=['object_ID'], inplace=True)  # 객체 ID 기준으로 정렬
    objLifespandf = reqdf.groupby(['object_ID']).agg({'request_time': ['min', 'max']})  # 객체별 최소/최대 요청 시간 계산
    objLifespandf.columns = ['_'.join(col).strip() for col in objLifespandf.columns.values]  # 열 이름 평탄화
    objLifespandf.rename(columns={'request_time_min': 'minRequestTime'}, inplace=True)  # 최소 요청 시간 열 이름 변경
    objLifespandf.rename(columns={'request_time_max': 'maxRequestTime'}, inplace=True)  # 최대 요청 시간 열 이름 변경
    objLifespandf['object_ID'] = objLifespandf.index  # 객체 ID를 인덱스에서 열로 이동
    objLifespandf['lifeSpan'] = (objLifespandf['maxRequestTime'] - objLifespandf['minRequestTime']) / 86400  # 수명 계산 (일 단위)
    min_request_time = reqdf['request_time'].min()  # 전체 요청 중 최소 요청 시간
    objLifespandf['start_day'] = (objLifespandf['minRequestTime'] - min_request_time) / 86400  # 도입 날짜 계산
    objLifespandf['end_day'] = (objLifespandf['maxRequestTime'] - min_request_time) / 86400  # 종료 날짜 계산
    objLifespandf["start_day"] = objLifespandf["start_day"].apply(int)  # 도입 날짜를 정수형으로 변환
    objLifespandf["end_day"] = objLifespandf["end_day"].apply(int)  # 종료 날짜를 정수형으로 변환
    objLifespandf.sort_values('start_day', ascending=True, inplace=True)  # 도입 날짜 기준으로 정렬
    objLifespandf.index.names = ['index']  # 인덱스 이름 설정

    # 객체 속성 파일 저장
    mergeddf = pd.merge(objfreqdf, objLifespandf, on='object_ID')  # 빈도 데이터와 수명 데이터를 병합
    mergeddf = mergeddf[['object_ID', 'frequency', 'lifeSpan', 'minRequestTime', 'maxRequestTime', 'start_day',
                         'end_day']]  # 필요한 열만 선택
    mergeddf.sort_values('start_day', ascending=True, inplace=True)  # 도입 날짜 기준으로 정렬
    mergeddf.to_csv(PROPERTIES_FILENAME, index=False)  # 속성 파일 저장

    print('속성 파일 저장 완료')
    del objfreqdf
    del objLifespandf
    del mergeddf

del reqdf  # 메모리 정리
print('완료')
