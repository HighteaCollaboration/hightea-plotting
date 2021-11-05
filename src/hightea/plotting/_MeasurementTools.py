import xml.etree.ElementTree as ET
import numpy as np

class MeasurementTools:
    files = []
    """stores all loaded xml files in the format: [filename XMLroot]"""
    def __init__(self):
        self.files = [];

    def nFiles(self):
        """return the number of loaded files"""
        return len(self.files)

    def printFileNames(self):
        """prints the names of included files"""
        if self.nFiles() == 0:
            print('No files imported')
        else:
            for f in self.files:
                print(f[0])

    def loadxml(self,filename,verbose=False):
        """
        takes a filename in form of a string and saves the XMLroot in 'files'.
        the ElementTree library is used here.
        """
        tree = ET.parse(filename)
        self.files.append([filename,tree.getroot()])
        if (verbose):
            print(f'MeasurementTools: loaded {filename}')

    def extractTree(self,fileid):
        """returns the XMLroot of file with the number 'fileid'"""
        return self.files[fileid][1]

    def extractSetups(self,fileid):
        """
        returns an array with strings describing the setups of the measurement
        'fileid'
        """
        xsecs = self.files[fileid][1].findall('xsection')
        setups = []
        for xs in xsecs:
          setups.append([xs.find('incoming').text,xs.find('scales').text])
        return setups

    def extractXSections(self,fileid):
        """
        returns the inclusive cross sections for all setups of measurement
        'fileid' as an expansion in ep in the format:
        [ [ xs_setup1 mcerr_setup1 ]
          [ xs_setup2 mcerr_setup2 ]
          ...                        ]
        where xs_setup is a lists with the coefficients of the ep expansion:
        if xs = sum_{i=-4}^0 c_i ep^i then
        xs_setup = [ c_0 c_-1 c_-2 c_-3 c_-4 ]
        similar for the errors
        """
        xsecs = self.files[fileid][1].findall('xsection')
        numxsecs = []
        for xs in xsecs:
          val = list(map(float,xs.find('values').text.split(',')))
          err = list(map(float,xs.find('errors').text.split(',')))
          numxsecs.append([val,err])
        return np.array(numxsecs)

    def extractObservables(self,fileid):
        """returns a list of observables names for measurement 'fileid'"""
        obs = self.files[fileid][1].findall('observable')
        unique = []
        for ob in obs:
            obsname = ob.find('description').text
            if (obsname in unique):
                continue
            else:
                unique.append(obsname)
        return unique

    def extractHistograms(self,fileid,obsname):
        """
        returns the list of all histograms of observable 'obsname' of
        measurement 'fileid'
        """
        obs = self.files[fileid][1].findall('observable')
        selectob = []
        for ob in obs:
          if ob.find('description').text == obsname :
            selectob.append(ob)
        return selectob

    @staticmethod
    def histogramEdges(hist):
        """
        takes as an argument a histogram in the format provided by
        extractHistograms() and returns the bin edges in the format:
        [ [dim1edge1 dim1edge2 ... dim1edgen ]
          [dim2edge1 dim2edge2 ... dim2edgen ]
          ...                                  ]
        as a list of lists
        """
        hs = hist.find('histogram')
        edges = hs.findall('edges')
        edgesList = []
        for edge in edges :
          edgesList.append(list(map(float,edge.text.split(','))))
        return edgesList

    @staticmethod
    def histogramSmearing(hist):
        """
        takes as an argument a histogram in the format provided by
        extractHistograms() and returns the smearing parameter:
        as floating number
        """
        hs = hist.find('histogram')
        return float(hs.find('smearing').text)

    @staticmethod
    def histogramBinWidths(hist):
        """
        takes as an argument a histogram in the format provided by
        extractHistograms() and returns the bin widths in the format:
        [ [dim1width1 dim1width2 ... dim1widthn ]
          [dim2width1 dim2width2 ... dim2widthn ]
          ...                                     ]
        as numpy array
        """
        edges = MeasurementTools.histogramEdges(hist)
        binwidths = []
        for jt in range(0,len(edges)) :
          curbinwidths = []
          for it in range(0,len(edges[jt])-1) :
            curbinwidths.append(edges[jt,it+1]-edges[jt,it])
          binwidths.append(curbinwidths)
        return np.array(binwidths)

    def histogramValues(self,hist,withOUF=False):
        """
        takes as an argument a histogram in the format provided by
        extractHistograms() and returns the bin values in the format
        of a array with the index structure
           val_{i_1,i_2,...,i_{#dim}}
        where val_{...} is given by a list of values:
        val_{...} = [ c_0^1 c_0^2 ... c_0^m c_-1^1 c_-1^2 ... c_-4^m ]
        where c_i^j is the ep^i coefficient of the integrated cross section
        in that bin for setup j (j = 1,...,m with m = numbers of setup).
        The order is the same as returned by extractSetups
        """
        hs = hist.find('histogram')
        bins = hs.findall('bin')
        edges = self.histogramEdges(hist) # to get the dimensions
        hdim = len(edges)
        vals = []
        for b in bins:
          vals.append(list(map(float,b.find('values').text.split(','))))

        vals = np.array(vals)
        #reshape the array
        newshape = []
        nentries = len(vals[0])
        for edge in edges :
          newshape.append(len(edge)+1) # account for OUF histograms
        newshape.append(nentries)
        vals = np.reshape(vals,newshape)

        if not withOUF : # remove OUF bins
          for it in range(0,len(edges)) :
            vals = np.delete(vals,0,it)
            vals = np.delete(vals,-1,it)
        return vals

    def histogramErrors(self,hist,withOUF=False):
        """
        takes as an argument a histogram in the format provided by
        extractHistograms() and returns the bin values in the format
        of a array with the index structure
           mcerr_{i_1,i_2,...,i_{#dim}}
        where mcerr_{...} is given by a list of values:
        mcerr_{...} = [ c_0^1 c_0^2 ... c_0^m c_-1^1 c_-1^2 ... c_-4^m ]
        where c_i^j is the ep^i coefficient of the mcerr of integrated
        cross section in that bin for setup j (j = 1,...,m with m = numbers of
        setup). The order is the same as returned by extractSetups
        """
        hs = hist.find('histogram')
        bins = hs.findall('bin')
        edges = self.histogramEdges(hist) # to get the dimensions
        hdim = len(edges)
        vals = []
        for b in bins:
          vals.append(list(map(float,b.find('errors').text.split(','))))

        vals = np.array(vals)
        #reshape the array
        newshape = []
        nentries = len(vals[0])
        for edge in edges :
          newshape.append(len(edge)+1) # account for OUF bins
        newshape.append(nentries)
        vals = np.reshape(vals,newshape)

        if not withOUF : # remove OUF bins
          for it in range(0,len(edges)) :
            vals = np.delete(vals,0,it)
            vals = np.delete(vals,-1,it)
        return vals

    def histogramHits(self,hist,withOUF=False):
        """
        takes as an argument a histogram in the format provided by
        extractHistograms() and returns the bin values in the format
        of a array with the index structure
           nhits_{i_1,i_2,...,i_{#dim}}
        where nhits_{...} is the number of hits
        """
        hs = hist.find('histogram')
        bins = hs.findall('bin')
        edges = self.histogramEdges(hist) # to get the dimensions
        hdim = len(edges)
        vals = []
        for b in bins:
          vals.append(int(b.find('nhits').text))

        vals = np.array(vals)
        #reshape the array
        newshape = []
        for edge in edges :
          newshape.append(len(edge)+1) # account for OUF histograms
        vals = np.reshape(vals,newshape)

        if not withOUF : # remove OUF bins
          for it in range(0,len(edges)) :
            vals = np.delete(vals,0,it)
            vals = np.delete(vals,-1,it)
        return vals
