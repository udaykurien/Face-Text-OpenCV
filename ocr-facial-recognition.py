# Import modules
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import pytesseract as PT
import cv2 as cv
from kraken import pageseg # Kraken downgraded to 2.0.1 due to 'Too many connected components for a page image' error'
import numpy as np
import os
from zipfile import ZipFile
# from IPython.display import display
from matplotlib import pyplot, image
import inspect
import time
import random
import kraken

# Global/Main variables
cwd = os.getcwd()
imgs_zip_dir = 'images.zip'
imgs_zip_path = os.path.join(cwd,imgs_zip_dir)

# Greyscale and Binarize images with pillow and numpy
def gray_and_binarize(pil_img):
    ''' Binarize pillow image, Return binarized image '''
    start = time.time()
    # GRAY SCALE CONVERSION
    pil_img_gry = pil_img.convert('L')
    # BINARIZATION
    # Convert image to numpy array (computations faster than pixelwise computation)
    np_img_gry = np.asarray(pil_img_gry)
    # Calculate average pixel value for first guess of binarization threshold
    avg_pxl_val = round(np.average(np_img_gry),0)
    threshold = avg_pxl_val
    # Filter numpy array based on threshold value
    np_bin_img = np.where(np_img_gry < threshold, 0, 255)
    # Convert numpy array back into pillow image
    pil_bin_img = Image.fromarray(np_bin_img.astype('uint8'))
    end = time.time()
    print('{} completed in {} seconds'.format('Binarization', round(end - start),2))
    return pil_bin_img

# Segment image with kraken and return text boundaries
def seg_img(pil_bin_img):
    ''' Perform default segmentation with kraken, Return bounding boxes '''
    start = time.time()
    seg_boundaries = kraken.pageseg.segment(pil_bin_img, black_colseps=True)
    end = time.time()
    print('{} segmentation completed in {} seconds'.format('Kraken', round(end - start),2))
    return seg_boundaries

# Calculate average char width using tesseract on text within the boundaries
def average_char_length(pil_bin_img, seg_boundaries):
    # Find average character length (pixels)
    ''' Logic: Use pytesseract to read image contained in the segments returned by kraken.
        length (bounding_box)/len(string) returns an approximate of the character width.
        Find the average of all such character widths '''
    start = time.time()
    seg_avg_char_wdth = 0
    sum_avg_char_wdth = 0
    avg_calc_counter = 0
    # Don't consider any return text < 6 char in calc of avg char width to filter mistakes
    char_len_filter = 6
    # Pad boxes so that words aren't cropped for pytesseract
    box_pad=5
    for i in seg_boundaries['boxes']:
        boundaries_pad=i.copy()
        boundaries_pad[0] -= box_pad   # x1
        boundaries_pad[1] -= box_pad   # y1
        boundaries_pad[2] += box_pad   # x2
        boundaries_pad[3] += box_pad   # y2
        # OCR from cropped segments of the image
        ocr_string=PT.image_to_string(pil_bin_img.crop(boundaries_pad))
        # Calculate char width from data filtered according to length
        if len(ocr_string) > char_len_filter:
            seg_avg_char_wdth = (i[2]-i[0])/len(ocr_string)
            sum_avg_char_wdth += seg_avg_char_wdth
            avg_calc_counter += 1
    avg_char_wdth = int(round(sum_avg_char_wdth/avg_calc_counter,0))
#         Test
#         print("Average char width of segment:{}\n\
#     Accumulated char widths:{}\n\
#     Number of segments included in calculation:{}"\
#             .format(seg_avg_char_wdth,sum_avg_char_wdth,avg_calc_counter))
#         display(pil_bin_tst_img.crop(boundaries_pad))        
#         print(ocr_string)
#         print(avg_char_wdth)
    end = time.time()
    print('{} calculated in {} seconds'.format('Average char width', round(end - start),2))
    return avg_char_wdth

# Draw vertical separators based on avg char width and avg char height
def mark_vertical_seperators(pil_bin_img, ht_thresh, wdth_thresh):
    # Mark vertical segments
    ''' For each segmentation boundary draw a vertical line of length wdth_thresh
        if there are no black pixels that lie in its path'''
    start = time.time()
    pil_bin_img_marked = pil_bin_img.copy()
    pil_bin_img_marked_obj=ImageDraw.ImageDraw(pil_bin_img_marked)
    # Windowed raster scan
    for h in range(0,pil_bin_img.height-ht_thresh,ht_thresh):
        for w in range(0,pil_bin_img.width-wdth_thresh,wdth_thresh):
            scan_window=[w,h,w+wdth_thresh,h+ht_thresh] #x1,y1 -> Top Left ,x2,y2 -> Bottom Right
            ''' Testing
            # Generate scan window grid
            debug_img_2_obj.rectangle(scan_window) '''
            blk_pxl_trckr = 0 # Check for black pixels in scan window
            for scan_window_h in range(scan_window[1],scan_window[3]):
                for scan_window_w in range(scan_window[0],scan_window[2]):
                    pxl_chk=pil_bin_img.getpixel((scan_window_w, scan_window_h))
                    ''' Testing
                    # Copy original image to check that raster scan is working
                    # debug_img.putpixel((scan_window_w,scan_window_h),pxl_chk)'''
                    if pxl_chk == 0:
                        blk_pxl_trckr += 1
            if blk_pxl_trckr == 0:
                # Draw vertical line in the middle of the scan window
                line_x = int((scan_window[0]+scan_window[2])/2)
                line_y1 = scan_window[1]
                line_y2 = scan_window[3]
                pil_bin_img_marked_obj.line((line_x, line_y1, line_x, line_y2))
    end = time.time()
    print('{} segmentation completed in {} seconds'.format('Vertical', round(end - start),2))
    return pil_bin_img_marked

#Main function for vertical segmentation of pillow image file
#Dependencies: seg_img, average_char_length, mark_vertical_seperators
def vert_segment(pil_bin_img):
    ''' Construct vertical seperator to aid OCR,
        Return binarized image with vertical seperators drawn '''
    print('Starting Kraken segmentation.')
    seg_boundaries = seg_img(pil_bin_img)
    # Calculate average height of line of text from segment boxes
    sum_box_ht = 0
    sum_box_wdth = 0
    for i in seg_boundaries['boxes']:
        sum_box_ht += i[3] - i[1]
        sum_box_wdth += i[2] - i[0]
    # print(seg_boundaries['boxes'])
    avg_box_ht = int(round(sum_box_ht/len(seg_boundaries['boxes']),0))
    avg_box_wdth = int(round(sum_box_wdth/len(seg_boundaries['boxes']),0))
    print('Calling average character width.')
    avg_char_wdth = average_char_length(pil_bin_img, seg_boundaries)
     # Define some threshold values for scanning window
    thresh_const_ht = 2
    thresh_const_wdth = 1.5
    ht_thresh = avg_box_ht * thresh_const_ht
    wdth_thresh = int(avg_char_wdth * thresh_const_wdth)
    print('Calling mark vertical seperators.')
    pil_bin_img_marked = mark_vertical_seperators(pil_bin_img,ht_thresh,wdth_thresh)
    return pil_bin_img_marked


#OpenCV face detection
def face_detect(pil_img):
    pil_img_drw_obj = ImageDraw.ImageDraw(pil_img)
    pil_img_gry=pil_img.convert('L')
    pil_img_enhnce_obj = ImageEnhance.Contrast(pil_img_gry)
    pil_img_gry = pil_img_enhnce_obj.enhance(2)
    np_img_gry = np.asarray(pil_img_gry)
    
    faceCascadeFrontal = cv.CascadeClassifier(cv.data.haarcascades + 
                                       "haarcascade_frontalface_default.xml")
    faces_frontal = faceCascadeFrontal.detectMultiScale(
        np_img_gry,
        scaleFactor=1.2,
        minNeighbors=8,
    )
    
    crop_img_lst = []
    for x,y,w,h in faces_frontal:
        crop_img = pil_img.crop((x,y,x+w,y+h))
        crop_img_lst.append(crop_img)
    
    return crop_img_lst

# Extract zip files to memory and perform text conversion on it
def text_from_zip(int_flag = 0):
    # 0: text extraction without segmentation
    # 1: text extraction with vertical segmentation using kraken 
    imgs_zipfile = ZipFile(imgs_zip_path,'r')
    inf_lst = imgs_zipfile.infolist()
    page_txt_dict = {}
    page_image_dict = {}
    counter = 1

    # Read all files and store text in page_txt_dict with info-list as keys
    for f in inf_lst:
        print("\nStarting operations on Page {} {}.".format(counter,f.filename)+'\n'+'->'*15)
        ifile = imgs_zipfile.open(f)
        pil_img = Image.open(ifile)
        pil_img_bin = gray_and_binarize(pil_img)
        if int_flag == 0:
            start = time.time()
            page_txt_dict[f] = PT.image_to_string(pil_img_bin)
            end = time.time()
            print('Text recognition completed in {} seconds'.format(round(end - start, 0)))
        elif int_flag == 1:
            page_txt_dict[f] = vert_segment(pil_img_bin)
        counter += 1
        
    return page_txt_dict



# Extract faces from pages containing text matching input pattern
# Dependencies: text_from_zip, face_detect, contact_sheet</i>
def match_faces(str_name, int_flag):
    imgs_zipfile = ZipFile(imgs_zip_path,'r')
    page_txt_dict = text_from_zip(int_flag)
    name_meta_data = {}
    # Search page_txt_dict for all instances of name and store the keys of matching pages
    # in new dictionary name_meta_data
    for k in page_txt_dict.keys():
        if str_name in page_txt_dict[k]:
            name_meta_data[k]=[]
    # Analyze pages referenced by keys of name_meta_data, with openCV, and store retrieved
    # images as a list to the keys to name_meta_data
    for k in name_meta_data.keys():
        tmp_lst = []
        ifile = imgs_zipfile.open(k)
        pil_img = Image.open(ifile)
        tmp_lst = face_detect(pil_img)
    #     if len(tmp_lst) > 0:
        # Unpack list received from face detect in list with appropriate key in name_meta_data
        for j in tmp_lst:
            name_meta_data[k].append(j)
    final_contact_sheet = contact_sheet(name_meta_data)
    return final_contact_sheet


# Composite matches into a contact sheet
def contact_sheet(name_meta_data):
    img_per_row = 5
    img_size = (600, 600)
    contct_wdth = img_size[0] * img_per_row
    txt_ht = 200
    # font = ImageFont.truetype('~/.fonts/fira-code/ttf/FiraCode-Bold.ttf',85)
    font = ImageFont.load_default()
    cntct_lst = []
    
    # Generate individual contact sheets and gather them in a list
    for k in name_meta_data:
        img_num = len(name_meta_data[k])
        if img_num > 0:
            contct_rows_num = img_num // 6
            contct_template = Image.new(mode='RGB', size=(contct_wdth,
                                                (img_size[0]*(contct_rows_num+1)+txt_ht)))
            draw_obj = ImageDraw.Draw(contct_template)
            draw_obj.rectangle((0,0,contct_template.size[0],txt_ht), fill='#ffffff')
            draw_obj.text((20, 40), "Results found in file {}".format(k.filename)
                          , '#000000', font)
            x=0
            y=txt_ht
            for img in name_meta_data[k]:
                img_res = img.resize(img_size)                 
                contct_template.paste(img_res,(x,y))
                if x + img_res.size[0] < contct_template.size[0]:
                    x += img_res.size[0]
                else:
                    x = 0
                    y += img_res.size[1]
            cntct_lst.append(contct_template)
        elif img_num == 0:
            contct_template = Image.new(mode='RGB', size=(contct_wdth,int(1.5*txt_ht)))
            draw_obj = ImageDraw.Draw(contct_template)
            draw_obj.rectangle((0,0,contct_template.size[0],2*txt_ht), fill='#ffffff')
            draw_obj.text((20, 40), "Results found in file {}".format(k.filename)
                          , '#000000', font)
            draw_obj.text((20, 185), "But there were no faces in that file!"
                          , '#000000', font)
            cntct_lst.append(contct_template)
            
    # Composite the sheets
    total_ht = 0
    for sheet in cntct_lst:
        total_ht += sheet.size[1]
            
    final_contact_sheet = Image.new(mode='RGB', size=(contct_wdth,total_ht))
    
    x = 0
    y = 0
    for sheet in cntct_lst:
        final_contact_sheet.paste(sheet,(x,y))
        y += sheet.size[1]
        
    return final_contact_sheet

# Test run
# if __name__ == "__main__":
start_1 =  time.time()
christopher = match_faces('Christopher',0)
end_1 = time.time()
runtime_1 = int(end_1-start_1)

start_2 = time.time()
mark = match_faces('Mark',0)
end_2 = time.time()
runtime_2 = int(end_2-start_2)

print("Total run time (Christopher) : {} seconds \n\
Total run time (Makr): {} seconds".format(runtime_1, runtime_2))
print('Christopher')
display(christopher)
print('Mark')
display(mark)

