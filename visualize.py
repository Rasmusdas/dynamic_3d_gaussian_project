import numpy as np
import torch
import matplotlib.pyplot as plt
from diff_gaussian_rasterization import GaussianRasterizer as Renderer, GaussianRasterizationSettings as Camera
import os
import cv2
import time
import math
from collections import namedtuple

w = 1280
h = 720
RENDER_BOTH = False


def setup_camera(w, h, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0,0,0], dtype=torch.float32, device="cuda"),
        scale_modifier=1,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
    return cam

def load_scene_data(file_path = "",seg_as_col=False,remove_background=False):
    params = dict(np.load(file_path))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D': params['means3D'][t],
            'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        if remove_background:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if remove_background:
        is_fg = is_fg[is_fg]
    return scene_data, is_fg


def load_scene_data_combined(params,seg_as_col=False,remove_background=False):
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D': params['means3D'][t],
            'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        if remove_background:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if remove_background:
        is_fg = is_fg[is_fg]
    return scene_data, is_fg

def render(w2c, k, timestep_data):
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, 0.1, 1000)

        cam_test = cam._asdict()

        cam_tuple = namedtuple('Cam', list(cam_test.keys()))
        cam = (cam_tuple(*list(cam_test.values())))

        im, _, depth, = Renderer(raster_settings=cam)(**timestep_data)
        return im, depth
    
def init_camera(x_angle=0,y_angle=0,z_angle=0, x_t = 0,y_t = 0, z_t = 0, f_ratio=0.82):
    r = make_rot_mat(x_angle,y_angle,z_angle)
    
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return r, k

def make_rot_mat(x_angle,y_angle,z_angle):
    
    ry = y_angle * np.pi / 180
    rx = x_angle * np.pi / 180
    rz = z_angle * np.pi / 180
    r1 = np.array([[1, 0., 0., 0.],
                    [0.,         np.cos(rx), -np.sin(rx),          0],
                    [0,          np.sin(rx), np.cos(rx),  0],
                    [0.,         0.,0.,         1.]])
    r2 = np.array([[np.cos(ry), 0., -np.sin(ry), -1.],
                    [0.,         1., 0.,          -2],
                    [np.sin(ry), 0., np.cos(ry),  0],
                    [0.,         0., 0.,          1.]])
    r3 = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz), np.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    return r1@r2@r3


def get_render_data_single_scene(param_path):
    #global start_x,start_y,i,j,move,l,d,move,t,cam_h,timeStep
    if RENDER_BOTH:
        fg_dat, is_fg = load_scene_data(param_path,remove_background=False)
        bg_dat, is_fg = load_scene_data(param_path,remove_background=True)
        return fg_dat,bg_dat
    else:
        data, is_fg = load_scene_data(param_path,remove_background=True)
        return data
        

def get_render_data_combined_scene(fg_path,bg_path, fg_rot_angs = [0,0,0]):
    
    print("Loading")
    data1 = np.load(fg_path)
    data2 = np.load(bg_path)

    n = min(len(data1["means3D"]),len(data2["means3D"]))
    data1 = dict(data1)
    rot_mat = make_rot_mat(*fg_rot_angs)
    for i in range(len(data1["logit_opacities"])):
        if data1["seg_colors"][i,2] == 1:
            data1["means3D"][:,i] = np.array([-100000,-100000,-100000])
            
        for t in range(0,81):
            data1["means3D"][t,i]=(rot_mat@np.hstack((data1["means3D"][t,i],[1])))[:3]
        
        if (i % 10000) == 0:
            print(f"{i}/{len(data1['logit_opacities'])}")

    print("Loaded FG")
    label_1 = ["means3D","rgb_colors","unnorm_rotations"]
    label_2 = ["logit_opacities","log_scales","seg_colors"]

    combined_data = {}

    for label in label_1:
        temporal_data = []
        for t in range(81):
            temporal_data.append(np.vstack((data1[label][t],data2[label][0])))   
            print(t)
        combined_data[label] = np.array(temporal_data)
        print(label)
        
    for label in label_2:
        combined_data[label] = np.vstack((data1[label],data2[label]))
        print(label)

    print("Loaded BG")
    data, is_fg = load_scene_data_combined(combined_data)
    
    return data
    
def render_scene(fg_dat=[],bg_dat=None):

    if bg_dat == None:
        data= fg_dat
        
    print("Loaded")
    start_x,start_y = 0,0
    timeStep = 0
    i = 0
    j = 0
    l = 0
    d = 1
    move = 0
    t = np.array([0.0,0.0,0.0,1])
    cam_h = 1
    r = np.eye(3,3)
    timeSteps = len(fg_dat)
    def generate_frames():
        nonlocal r
        while True:
            r, k = init_camera(x_angle=j,y_angle=-i,z_angle=l,x_t=0,y_t=0,z_t=0)
            r[:,3] = t
            if RENDER_BOTH:
                fg_img, depth = render(r,k,fg_dat[timeStep])
                bg_img, depth = render(r,k,bg_dat[timeStep])
                yield fg_img,bg_img
            else:
                im, depth = render(r,k,data[timeStep])
                yield im,[]
            
    def on_click(event,x,y,flags,param):
        nonlocal start_x,start_y,i,j,move
        if event == cv2.EVENT_LBUTTONDOWN:
            start_x,start_y = x,y
            move = True
        if event == cv2.EVENT_LBUTTONUP:
            move = False
            
        if move:
            i-=(x-start_x)/25
            j+=(y-start_y)/25
            
            start_x,start_y = x,y
    end = 0
    doTime = 0

    start = time.time()

    while True:
        for fg,bg in generate_frames():
            
            if RENDER_BOTH:
                im_fg_np = fg.detach().cpu().numpy()
                im_fg_np = np.swapaxes(im_fg_np,0,1)
                im_fg_np = np.swapaxes(im_fg_np,1,2)
                
                im_bg_np = bg.detach().cpu().numpy()
                im_bg_np = np.swapaxes(im_bg_np,0,1)
                im_bg_np = np.swapaxes(im_bg_np,1,2)           
                im_np = np.hstack((im_fg_np,im_bg_np))
            
            else:
                im_np = fg.detach().cpu().numpy()
                im_np = np.swapaxes(im_np,0,1)
                im_np = np.swapaxes(im_np,1,2)
                
            cv2.namedWindow('image')
            cv2.setMouseCallback('image',on_click)
            
            if doTime:
                timeStep = (timeStep+1) % timeSteps
            
            im_np = cv2.cvtColor(im_np,cv2.COLOR_BGR2RGB)
            
            cv2.imshow('image', im_np)
            k = cv2.waitKey(17)
            start = time.time()
            if k == 120:
                timeStep = (timeStep+1) % timeSteps
            if k == 122:
                timeStep = (timeStep-1) % timeSteps
                
            if k == 116:
                doTime = not doTime
            if k == 97:
                t+= r@np.array([0.1,0,0,0])
            if k == 100:
                t-= r@np.array([0.1,0,0,0])
            if k == 119:
                t-= r@np.array([0,0,0.1,0])
            if k == 115:
                t+= r@np.array([0,0,0.1,0])
            if k == 49:
                t-= r@np.array([0,0.1,0,0])
            if k == 50:
                t+= r@np.array([0,0.1,0,0])
                
            if k == 113:
                d -= 0.1
            if k == 101:
                d += 0.1
                
            if k == 102:
                cam_h -= 0.1
            if k == 114:
                cam_h+= 0.1
                
            if k == 27:
                break
            
        
if __name__ == "__main__":
    RENDER_BOTH = False
    dat = get_render_data_combined_scene("params_fg.npz","max_params.npz")
    
    render_scene(dat)