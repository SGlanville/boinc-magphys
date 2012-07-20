import pyfits
from PIL import Image, ImageDraw
import math
import sys
import os
from database.database_support import Galaxy, Area, AreaUser, PixelResult, PixelUser, login
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, aliased

class FitsImage:
    def __init__(self):
        #fitsimage.FitsImage.__init__(self)
        pass
    
    def buildImage(self, fitsFileName, imageDirName, imagePrefixName, method, createBWImages, createLog, debug):
        if imageDirName[-1] != "/":
            imageDirName = imageDirName + "/"
        if os.path.isfile(imageDirName):
            print 'Directory ', imageDirName , 'exists'
            return 1
        elif os.path.isdir(imageDirName):
            pass
        else:
            os.mkdir(imageDirName)
        
        hdulist = pyfits.open(fitsFileName, memmap=True)
        if debug:
            hdulist.info()
        
        blackRGB = (0, 0, 0)
        black = (0)
        white = (255)
        
        hdu = hdulist[0]
        width = hdu.header['NAXIS1']
        height = hdu.header['NAXIS2']
        
        #cards = ['MAGPHYSL', 'MAGPHYSN', 'MAGPHYSI', 'MAGPHYSF']
        cards = ['MAGPHYSL', 'MAGPHYSN', 'MAGPHYSI', 'MAGPHYSF', 'BITPIX', 'NAXIS', 'NAXIS2', 'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'BZERO', 'BSCALE', 'OBJECT']
        
        # Create Three Colour Images
        image1Filters = [118, 117, 116] # i, r, g
        image2Filters = [117, 116, 124] # r, g, NUV
        image3Filters = [280, 116, 124] # 3.6, g, NUV
        image4Filters = [283, 117, 124] # 22, r, NUV
        
        image1 = Image.new("RGB", (width, height), blackRGB)
        image2 = Image.new("RGB", (width, height), blackRGB)
        image3 = Image.new("RGB", (width, height), blackRGB)
        image4 = Image.new("RGB", (width, height), blackRGB)
        
        images = [image1, image2, image3, image4]
        imageFilters = [image1Filters, image2Filters, image3Filters, image4Filters]
        
        file = 0
        loCut = 0
        hiCut = 999999
        for hdu in hdulist:
            file = file + 1
            if debug:
                print hdu,
                print file
            
            width = hdu.header['NAXIS1']
            height = hdu.header['NAXIS2']
            filter = hdu.header['MAGPHYSI']
              
            # Get the Three Colour Images that require this filter.
            rImages = []
            gImages = []
            bImages = []
            for idx in range(len(imageFilters)):
                img = images[idx]
                filters = imageFilters[idx]
                if filters[0] == filter:
                    rImages.append(img)
                if filters[1] == filter:
                    gImages.append(img)
                if filters[2] == filter:
                    bImages.append(img)
            
            xoffset = 0
            yoffset = 0        
            minorigvalue = 99999999.0
            maxorigvalue = -99999999.0
            minvalue = 9999999.0
            maxvalue = -99999999.0
            zerocount = 0
            sumsq = 0
            sum = 0
            count = 0
            for x in range(0, width-1):
                for y in range(0, height-1):
                    value =  hdu.data[y + xoffset,x + xoffset]
                    if not math.isnan(value):
                        if value < minorigvalue:
                            minorigvalue = value
                        if value > maxorigvalue:
                            maxorigvalue = value
                        #if value < 0:
                        #    value = 0
                        if value == 0:
                            zerocount = zerocount + 1
                            continue
                        if method == 'atan':
                            value = math.atan(value)
                        elif method == 'log':
                            #if value < 0.001:
                            #    value = 0.001;
                            value = math.log(value)
                        elif method == 'log10':
                            #if value < 0.001:
                            #    value = 0.001;
                            value = math.log(value, 10)
                        elif method == 'linear':
                            if value < 0.0:
                                value = 0.0
                            #if value > math.pi/2:
                            #    value = math.pi/2
                                
                        sum = sum + value
                        sumsq = sumsq + (value*value)
                        count = count + 1
                        if value < minvalue:
                            minvalue = value
                        if value > maxvalue:
                            maxvalue = value
            avg = sum / count
            if debug:
                 print sumsq, count, avg
            stddev=0.0
            try:
                stddev = math.sqrt((sumsq/count) - (avg*avg))
            except ValueError:
                stddev=0.0
            
            valuerange = []
            for z in range(0, 256):
                valuerange.append(0)
            if method == 'linear':
                minvalue = 0.0
                hiCut = avg + (15*stddev)
                if hiCut > maxvalue:
                    maxvalue = hiCut
                
            mult = 256.0 / (maxvalue - minvalue)
            sminvalue = 99999999
            smaxvalue = 0
            pixelcount = 0
                
            if debug:
                print 'Min-Orig', minorigvalue, 'Max-Orig', maxorigvalue, 'Min-Adj', minvalue, 'Max-Adj', maxvalue, 'Multiplier', mult, 'Zeroes', zerocount
            
            if createBWImages:
                imagebw = Image.new("L", (width, height), black)
                imagewb = Image.new("L", (width, height), white)
            
            for x in range(0, width-1):
                for y in range(0, height-1):
                    value =  hdu.data[y + xoffset,x + xoffset]
                    if not math.isnan(value):
                        #if value < 0:
                        #    value = 0
                        if method == 'atan':
                            value = math.atan(value)
                        elif method == 'log':
                            if value == 0:
                                continue
                            #if value < 0.001:
                            #    value = 0.001;
                            value = math.log(value)
                        elif method == 'log10':
                            if value == 0:
                                continue
                            #if value < 0.001:
                            #    value = 0.001;
                            value = math.log(value, 10)
                        elif method == 'linear':
                            if value < 0.0:
                                value = 0.0
                        adjvalue = value - minvalue
                        value = int(adjvalue*mult)
                        #value = int(value*256)
                        if value < 0:
                            value = 0
                        if value > 255:
                            value = 255
                        #print math.atan(value)
                        if value < sminvalue:
                            sminvalue = value
                        if value > smaxvalue:
                            smaxvalue = value
                        valuerange[value] = valuerange[value] + 1
                        #print 'x', x, 'y', width-y-1, 'Value', value, 'Width', width, 'Height', height
                        if createBWImages:
                            imagebw.putpixel((x,width-y-1), (value))
                            imagewb.putpixel((x,width-y-1), (255-value))
                        if value > 0:
                            pixelcount = pixelcount + 1
                        for img in rImages:
                            px = img.getpixel((x,width-y-1))
                            img.putpixel((x,width-y-1), (value, px[1], px[2]))
                        for img in gImages:
                            px = img.getpixel((x,width-y-1))
                            img.putpixel((x,width-y-1), (px[0], value, px[2]))
                        for img in bImages:
                            px = img.getpixel((x,width-y-1))
                            img.putpixel((x,width-y-1), (px[0], px[1], value))
            
            if debug:
                print 'Scaled: Min', sminvalue, 'Max', smaxvalue
                print 'Pixel Count', pixelcount
            #print 'Count', count, 'Avg', avg, 'Std Dev', stddev
            #print 'LoCut', avg - (10*stddev), 'HiCut', avg + (15*stddev)
            
            if createLog:
                logFile = open(imageDirName + imagePrefixName + '_' + str(file) + '.txt', 'w')
                self.printCardsToFile(logFile, hdu.header, cards)
                logFile.write('\n')
                logFile.write('Count {} Avg {} StdDev {} Pixels {} PercentWithValue {}\n'.format(count, avg, stddev, width*height, ((count-zerocount)*100.0)/(width*height)))
                logFile.write('Possible LoCut {} HiCut {}\n'.format(avg - (10*stddev), avg + (15*stddev)))
                logFile.write('Min-Orig {} Max-Orig {} Min-Adj {} Max-Adj {} Multiplier {} Zeroes {}\n'.format(minorigvalue, maxorigvalue, minvalue, maxvalue, mult, zerocount))
                logFile.write('\n')
            
                for z in range(0, 256):
                    logFile.write('{0:3d} {1}\n'.format(z, valuerange[z]))
                logFile.close()
            if createBWImages:
                imagebw.save(imageDirName + imagePrefixName + '_' + str(file) + '_bw.jpg')
                imagewb.save(imageDirName + imagePrefixName + '_' + str(file) + '_wb.jpg')
        
        image1.save(imageDirName + imagePrefixName + "_colour_1.jpg")
        image2.save(imageDirName + imagePrefixName + "_colour_2.jpg")
        image3.save(imageDirName + imagePrefixName + "_colour_3.jpg")
        image4.save(imageDirName + imagePrefixName + "_colour_4.jpg")
        
        hdulist.close()
    
    def markImage(self, inImageFileName, outImageFileName, galaxy_id, userid):
        engine = create_engine(login)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        image = Image.open(inImageFileName, "r").convert("RGBA")
        
        #pixels = session.query(PixelResult).filter("galaxy_id=:galaxyId", "user_id=:userId").params(galaxyId=galaxyId).all()
        areas = session.query(Area, AreaUser).filter(AreaUser.userid == userid)\
          .filter(Area.area_id == AreaUser.area_id)\
          .order_by(Area.top_x, Area.top_y).all()
        print 'Areas', len(areas)
        for areax in areas:
            area = areax.Area;
            for x in range(area.top_x, area.bottom_x):
                for y in range(area.top_y, area.bottom_y):
                    self.markPixel(image, x, y)
            #print px, px.x, px.y
            #x = int(px.x)
            #y = int(px.y)
            #self.markPixel(image, x, y)
        
        for x in range(140, 145):
            for y in range(80, 93):
                self.markPixel(image, x, y)
        
        for x in range(120, 125):
            for y in range(80, 86):
                self.markPixel(image, x, y)
        
        image.save(outImageFileName)
    
    def markPixel(self, image, x, y):
        px = image.getpixel((x,y))
        image.putpixel((x,y), (255,255,255))
        #image.putpixel((x,y), (px[0], px[1], px[2], 50))
        r = px[0] * 2
        g = px[1] * 2
        b = px[2] * 2
        if r > 255:
            r = 255
        if g > 255:
            g = 255
        if b > 255:
            b = 255
        if r < 20:
            r = 128
        if g < 20:
            g = 128
        if b < 20:
            b = 128
        #image.putpixel((x,y), (r, g, b))
        
    def userGalaxy(self, userid):
        engine = create_engine(login)
        Session = sessionmaker(bind=engine)
        session = Session()
        stmt = session.query(Galaxy.galaxy_id).join(Area).join(AreaUser).filter(AreaUser.userid == userid).subquery()
        #print stmt
        #print session.query(Galaxy).filter(Galaxy.galaxy_id.in_(stmt))
        #adalias = aliased(PixelResult, stmt);
        galaxyIds = []
        for galaxy in session.query(Galaxy).filter(Galaxy.galaxy_id.in_(stmt)):
           print 'Galaxy', galaxy.name
           galaxyIds.append(galaxy.galaxy_id)
        return galaxyIds;
        
    def printCardsToFile(self, outFile, header, keys):
        for key in keys:
            try:
                outFile.write('{0:8} {1}\n'.format(key, header[key]))
            except KeyError as e:
                pass
            
