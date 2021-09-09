import torch.nn.functional as F
import torch

def get_loss(net_output, ground_truth):
    criteria = torch.nn.BCEWithLogitsLoss()

    age_loss = criteria(net_output['age'], torch.stack(ground_truth['age']).permute(-1, 0).type_as(net_output['age']))
    mood_loss = criteria(net_output['mood'], torch.stack(ground_truth['mood']).permute(-1, 0).type_as(net_output['mood']))
    style_loss = criteria(net_output['style'], torch.stack(ground_truth['style']).permute(-1, 0).type_as(net_output['style']))
    time_loss = criteria(net_output['time'], torch.stack(ground_truth['time']).permute(-1, 0).type_as(net_output['time']))
    pose_loss = criteria(net_output['pose'], torch.stack(ground_truth['pose']).permute(-1, 0).type_as(net_output['pose']))
    setting_loss = criteria(net_output['setting'], torch.stack(ground_truth['setting']).permute(-1, 0).type_as(net_output['setting']))
    weather_loss = criteria(net_output['weather'], torch.stack(ground_truth['weather']).permute(-1, 0).type_as(net_output['weather']))
    face_loss = criteria(net_output['face'], torch.stack(ground_truth['face']).permute(-1, 0).type_as(net_output['face']))
    gender_loss = criteria(net_output['gender'], torch.stack(ground_truth['gender']).permute(-1, 0).type_as(net_output['gender']))
    clothing_loss = criteria(net_output['clothing'], torch.stack(ground_truth['clothing']).permute(-1, 0).type_as(net_output['clothing']))
    hair_loss = criteria(net_output['hair'], torch.stack(ground_truth['hair']).permute(-1,0).type_as(net_output['hair']))
    # loss = (hair_loss)/1
    loss = age_loss + mood_loss + style_loss + time_loss + pose_loss + setting_loss + weather_loss + face_loss + gender_loss + clothing_loss + hair_loss

    return loss
