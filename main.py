import cv2, sys
import matplotlib.pyplot as plt
import numpy as np

def mission01():
    # 이미지 읽기
    org = cv2.imread('mission/01.png', cv2.IMREAD_COLOR)
    ans = cv2.imread('mission/mission_image01.png', cv2.IMREAD_COLOR)
    
    # medianBlur를 통해서도 소금-후추 노이즈를 처리할수없었기에
    # 전체적인 노이즈 패턴을 참고하여 노이즈를 제거하기위해 해당 Denoising 적용
    res = cv2.fastNlMeansDenoisingColored(org, None, 6, 10, 7, 21)
    
    # Denoising을 통해서 흐려진 부분을 어느정도 샤프닝
    # 가우시안 블러를 이용하여 블러링 된 이미지 생성
    blurred_img = cv2.GaussianBlur(res, (5,5), 0) 

    # Unsharp Mask 생성 (원본 - 블러)
    unsharp_mask = cv2.subtract(res, blurred_img)

    # Unsharp Mask를 원본 이미지에 더하여 샤프닝 적용
    res = cv2.addWeighted(res, 0.6, unsharp_mask, 1, 0)
    
    # 이미지 출력
    cv2.namedWindow('org')
    cv2.moveWindow('org', 100, 100)
    cv2.imshow('org', org)
    
    cv2.namedWindow('res')
    cv2.moveWindow('res', 1200, 650)
    cv2.imshow('res', res)
    
    cv2.namedWindow('ans')
    cv2.moveWindow('ans', 100, 650)
    cv2.imshow('ans', ans)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def mission03():
    # 이미지 읽기
    org = cv2.imread('mission/03.png', cv2.IMREAD_COLOR)
    ans = cv2.imread('mission/mission_image03.png', cv2.IMREAD_COLOR)

    # 하늘 부드럽게 하기 위한 Denoising
    res = cv2.fastNlMeansDenoisingColored(org, None, 8, 10, 7, 21)
    
    # 이미지 출력
    cv2.namedWindow('org')
    cv2.moveWindow('org', 100, 100)
    cv2.imshow('org', org)
    
    cv2.namedWindow('res')
    cv2.moveWindow('res', 1200, 650)
    cv2.imshow('res', res)
    
    cv2.namedWindow('ans')
    cv2.moveWindow('ans', 100, 650)
    cv2.imshow('ans', ans)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def mission05():
    # 이미지 읽기
    org = cv2.imread('mission/05.png', cv2.IMREAD_COLOR)
    ans = cv2.imread('mission/mission_image05.png', cv2.IMREAD_COLOR)

    # 정답 이미지가 좀 더 선명해진듯하고, 원본의 비중은 낮췄습니다.
    # 픽셀의 B 값의 비중이 좀 더 높아진듯해서, HSV에서 채도 값을 증폭 시켰습니다.
    
    # 이미지 블러링
    blurred_img = cv2.GaussianBlur(org, (5, 5), 1)

    # Unsharp Mask 생성
    unsharp_mask = cv2.subtract(org, blurred_img)

    # 마스크를 원본 이미지에 더하기 (선명도 조절 가능)
    res = cv2.addWeighted(org, 0.6, unsharp_mask, 1, 0)
    
    # RGB를 HSV로 변환
    res = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)

    # Saturation (채도) 값 증폭 (값 범위: 0~255)
    res[:,:,1] = np.clip(res[:,:,1] * 2, 0, 255).astype(np.uint8) 

    # HSV를 다시 RGB로 변환
    res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    
    # 이미지 출력
    cv2.namedWindow('org')
    cv2.moveWindow('org', 100, 100)
    cv2.imshow('org', org)
    
    cv2.namedWindow('res')
    cv2.moveWindow('res', 1200, 650)
    cv2.imshow('res', res)
    
    cv2.namedWindow('ans')
    cv2.moveWindow('ans', 100, 650)
    cv2.imshow('ans', ans)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    mission01()
    mission03()
    mission05()