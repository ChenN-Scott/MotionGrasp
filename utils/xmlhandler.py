__author__ = 'Minghao Gou'
__version__ = '1.0'

from xml.etree.ElementTree import Element, SubElement, tostring
import xml.etree.ElementTree as ET
import xml.dom.minidom
from transforms3d.quaternions import mat2quat, quat2axangle
from transforms3d.euler import quat2euler, quat2mat
import numpy as np
import os

class xmlReader():
    def __init__(self, xmlfilename):
        self.xmlfilename = xmlfilename
        etree = ET.parse(self.xmlfilename)
        self.top = etree.getroot()

    def showinfo(self):
        print('Resumed object(s) already stored in '+self.xmlfilename+':')
        for i in range(len(self.top)):
            print(self.top[i][1].text)

    def gettop(self):
        return self.top

    def getposevectorlist(self):
        # posevector foramat: [objectid,x,y,z,alpha,beta,gamma]
        posevectorlist = []
        for i in range(len(self.top)):
            objectid = int(self.top[i][0].text)
            objectname = self.top[i][1].text
            objectpath = self.top[i][2].text
            translationtext = self.top[i][3].text.split()
            translation = []
            for text in translationtext:
                translation.append(float(text))
            quattext = self.top[i][4].text.split()
            quat = []
            for text in quattext:
                quat.append(float(text))
            quat_ = [quat[3],quat[0],quat[1],quat[2]]
            alpha, beta, gamma = quat2euler(quat_)
            x, y, z = translation
            alpha *= (180.0 / np.pi)
            beta *= (180.0 / np.pi)
            gamma *= (180.0 / np.pi)
            posevectorlist.append([objectid, x, y, z, alpha, beta, gamma])
        return posevectorlist

    def get_conveyor_info(self):
        posevectorlist = []
        for i in range(len(self.top)):
            translationtext = self.top[i][0].text.split()
            translation = []
            for text in translationtext:
                translation.append(float(text))
            quattext = self.top[i][1].text.split()
            quat = []
            for text in quattext:
                quat.append(float(text))
            quat_ = [quat[3],quat[0],quat[1],quat[2]]
            alpha, beta, gamma = quat2euler(quat_)
            x, y, z = translation
            alpha *= (180.0 / np.pi)
            beta *= (180.0 / np.pi)
            gamma *= (180.0 / np.pi)
            posevectorlist.append([x, y, z, alpha, beta, gamma])
        return posevectorlist

def empty_pose_vector(objectid):
    # [object id,x,y,z,alpha,beta,gamma]
    # alpha, beta and gamma are in degree
	return [objectid, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0]


def empty_pose_vector_list(objectidlist):
	pose_vector_list = []
	for id in objectidlist:
		pose_vector_list.append(empty_pose_vector(id))
	return pose_vector_list


def getposevectorlist(objectidlist, is_resume, num_frame, frame_number, xml_dir):
    if not is_resume or (not os.path.exists(os.path.join(xml_dir, '%04d.xml' % num_frame))):
        print('log:create empty pose vector list')
        return empty_pose_vector_list(objectidlist)
    else:
        print('log:resume pose vector from ' +
              os.path.join(xml_dir, '%04d.xml' % num_frame))
        xmlfile = os.path.join(xml_dir, '%04d.xml' % num_frame)
        mainxmlReader = xmlReader(xmlfile)
        xmlposevectorlist = mainxmlReader.getposevectorlist()
        posevectorlist = []
        for objectid in objectidlist:
            posevector = [objectid, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for xmlposevector in xmlposevectorlist:
                if xmlposevector[0] == objectid:
                    posevector = xmlposevector
            posevectorlist.append(posevector)
        return posevectorlist


def getframeposevectorlist(objectidlist, is_resume, frame_number, xml_dir):
    frameposevectorlist = []
    for num_frame in range(frame_number):
        if not is_resume or (not os.path.exists(os.path.join(xml_dir,'%04d.xml' % num_frame))):
            posevectorlist=getposevectorlist(objectidlist,False,num_frame,frame_number,xml_dir)	
        else:
            posevectorlist=getposevectorlist(objectidlist,True,num_frame,frame_number,xml_dir)
        frameposevectorlist.append(posevectorlist)
    return frameposevectorlist
