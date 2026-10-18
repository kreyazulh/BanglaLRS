import torch, cv2
roi = torch.load("/home/tigerit/Documents/banglavsr-asr/data/roi/-P6a2jCWs80_0466_0015169.pt")   # [T,96,96] uint8
for i in range(0, len(roi), max(1, len(roi)//5)):
    cv2.imwrite(f"check_{i}.png", roi[i].numpy())