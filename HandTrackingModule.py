import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode=False,maxHands = 2,detectionCon = 0.5,trackingCon = 0.5 ):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        
        self.mp_hands = mp.solutions.hands
        self.hand = self.mp_hands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackingCon)
        self. mp_draw = mp.solutions.drawing_utils 
        
    def findHands(self ,img,draw = True):   
        img_rgb = cv.cvtColor(img,cv.COLOR_BGRA2RGB)
        self.results = self.hand.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_lm in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img,hand_lm,self.mp_hands.HAND_CONNECTIONS)       
        
        return img      
    
    def findPosition(self, img, handNo=0, draw=True): 
        
        lm_list = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
        
            for id,lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w),int(lm.y*h)
                print(id, cx,cy)
                lm_list.append([id,cx,cy])
                if draw:
                    cv.circle(img, (cx,cy), 10, (255,0,255), cv.FILLED)
            
        return lm_list
    
def main():
    prev_time = 0
    curr_time = 0
    cap = cv.VideoCapture(0)
    detector = handDetector()


    while True:
        ret,img = cap.read()
        img = detector.findHands(img)
        
        lm_list = detector.findPosition(img)
        if len(lm_list) != 0:
            print(lm_list[4])
    
        # frame rate
        curr_time = time.time()
        fps = 1/(curr_time-prev_time)
        prev_time = curr_time
        
        cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_TRIPLEX,3,(255,0,255),3)
        
        cv.imshow("Image",img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == '__main__':
    main()


