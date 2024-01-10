import cv2
import MathNet as mnt

(H, W) = (400, 400)
class SlidingWindowObjectDetection():
    def __init__(self, pretrained_classifier_path, kwargs):
        self.model = mnt.MathNet()
        self.model.load_state_dict(torch.load(pretrained_classifier_path))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.kwargs = kwargs
    
    def test(self, image, step, ws):
        potential = []
        for y in range(0, image.shape[0] - ws[1], step):
            for x in range(0, image.shape[1] - ws[0], step):
                crop_img = image[y:y+28, x:x+28]
                #print(type(image), type(crop_img))
                crop_tensor = transforms.ToTensor()
                
               
                
                thresh = 120
                ret,thresh_img = cv2.threshold(crop_img, thresh, 255, cv2.THRESH_BINARY)

                #find contours
                contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_contours = np.uint8(np.zeros((crop_img.shape[0],crop_img.shape[1])))
                cv2.drawContours(img_contours, contours, -1, (255,255,255), 1)
                
                if (len(contours) > 1):
                    print(x,y, len(contours))
                    img = Image.fromarray(crop_img.astype('uint8'))
                    #img = Image.fromarray(img_contours.astype('uint8'))
                    display(img)
                    potential.append(crop_img)
        return potential
    
    def predict(self, lst):
        res = []
        for image in lst:
            img = Image.fromarray(image.astype('uint8'))
            convert_tensor = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.Grayscale(1),
                transforms.ToTensor()

            ])

            
            x_image = convert_tensor(img)
            x_image = x_image.unsqueeze(0).float()
            x_image = x_image.to(device)

            preds = self.model(x_image) 
            prob = preds.max()
            if prob >= self.kwargs['MIN_CONF']:
                print(prob.item(), (map_pred(preds.argmax()), preds))
                img = Image.fromarray(image.astype('uint8'))
                display(img)
                cv2.waitKey(1000)
            
                res.append((map_pred(preds.argmax()), preds))

        
    def visualize_rois(self, rois):
        fig, axes = plt.subplots(1, len(rois), figsize=(20, 6))
        for ax, roi in zip(axes, rois):
            ax.imshow(roi, cmap='gray')

    
    def get_preds(self, rois, locs):
        rois = np.array(rois, dtype="float32")
        preds = self.predict(rois)
        #preds = list(zip(preds.argmax(axis=1).tolist(), preds.max(axis=1).tolist()))
        res = []
        for i in range(0, len(preds)):
            res.append((map_pred(preds[i].argmax()), preds[i].max()))
        #print(res)
        labels = {}

        for (i, p) in enumerate(res):
            (label, prob) = p
            if prob >= self.kwargs['MIN_CONF']:
                box = locs[i]
                L = labels.get(label, [])
                L.append((box, prob))
                labels[label] = L
        return preds, labels
    
    def __call__(self, img):
        potential = self.test(img, self.kwargs['WIN_STEP'], self.kwargs['ROI_SIZE'])
        self.predict(potential)
        
        #rois, locs = self.get_rois_and_locs()
        
        #preds, labels = self.get_preds(rois, locs)
        #nms_labels = self.apply_nms(labels)
        #if self.kwargs['VISUALIZE']:
        #self.visualize_preds(img, nms_labels)
        return 1



def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# Возврат в набор результатов скользящего окна, временно не использованный
def get_slice(image, stepSize, windowSize):
    slice_sets = []
    for (x, y, window) in sliding_window(image, stepSize, windowSize):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        slice = image[y:y + winH, x:x + winW]
        slice_sets.append(slice)
    return slice_sets


if __name__ == '__main__':
    image = cv2.imread('TEST//5.jpg')

         # Настройка размера скользящего окна
    (winW, winH) = (224,224)
         # Размер шага
    stepSize = 100
    cnt = 0
    for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        # since we do not have a classifier, we'll just draw the window
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1000)

        slice = image[y:y+winH,x:x+winW]
        cv2.namedWindow('sliding_slice',0)
        cv2.imshow('sliding_slice', slice)
        cv2.waitKey(1000)
        cnt = cnt + 1

