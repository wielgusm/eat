
def plt_all_triangles_CP_CPdiff(brr,bll,sour,snr_cut=0.,saveF=False,add_text=''):
    fmtL = ['bo','ro','go','co','mo','ko','b^','r^','g^','c^','m^','k^','bv','rv','gv','cv','mv','kv']    
    brr = brr[brr['snr'] > snr_cut]; bll = bll[bll['snr'] >snr_cut]
    brr = brr[brr.source==sour]
    bll = bll[bll.source==sour]
    brr, bll = match_2_dataframes(brr, bll, 'triangle')
    if np.shape(brr)[0]>2.:
        AllTri = sorted(list(set(brr.triangle)))
        plt.figure(figsize=(10,10))
        
        for cou in range(len(AllTri)):
            brrTRI = brr[brr.triangle==AllTri[cou]]
            bllTRI = bll[bll.triangle==AllTri[cou]]
            plt.errorbar(np.mod(brrTRI.cphase,360), phase_diff(brrTRI.cphase,bllTRI.cphase)-np.mod(brrTRI.cphase,360),np.sqrt(np.asarray(brrTRI.sigmaCP)**2 +np.asarray(bllTRI.sigmaCP)**2),fmt=fmtL[cou],label = AllTri[cou],markersize='10')
            
        plt.xlabel('closure phase RR',fontsize=15)
        plt.ylabel('closure phases RR-LL difference',fontsize=15)
        plt.title(sour+', snr cut: '+str(snr_cut)+' '+add_text,fontsize=15)
        x1,x2,y1,y2 = plt.axis()
        plt.axis((-20,380,y1,y2))
        plt.legend()
        plt.axhline(y=.0,color='k',linestyle='--')
        plt.grid()
        plt.tight_layout()
        
        if saveF==True:
            if snr_cut > 0.:
                nameF = 'CPRR_CPdiff_'+sour+'_snr_'+str(int(snr_cut))+'_'+add_text+'.pdf'
            else:
                nameF = 'CPRR_CPdiff_'+sour+'_'+add_text+'.pdf'
            plt.savefig(nameF)
        plt.show()
        
def plt_all_triangles_CP_CP(brr,bll,sour,snr_cut=0.,saveF=False,add_text=''):
    fmtL = ['bo','ro','go','co','mo','ko','b^','r^','g^','c^','m^','k^','bv','rv','gv','cv','mv','kv']    
    brr = brr[brr['snr'] > snr_cut]; bll = bll[bll['snr'] >snr_cut]
    brr = brr[brr.source==sour]
    bll = bll[bll.source==sour]
    brr, bll = match_2_dataframes(brr, bll, 'triangle')
    if np.shape(brr)[0]>2.:
        AllTri = sorted(list(set(brr.triangle)))
        plt.figure(figsize=(10,10))
        
        for cou in range(len(AllTri)):
            brrTRI = brr[brr.triangle==AllTri[cou]]
            bllTRI = bll[bll.triangle==AllTri[cou]]
            xpl = np.asarray(np.mod(brrTRI.cphase,360))
            ypl = np.asarray(np.mod(bllTRI.cphase,360))
            xpl = xpl-360.*(xpl>180.)
            ypl = ypl-360.*(ypl>180.)
            #plt.errorbar(np.mod(brrTRI.cphase,360), np.mod(bllTRI.cphase,360),np.sqrt(np.asarray(brrTRI.sigmaCP)**2 +np.asarray(bllTRI.sigmaCP)**2),fmt=fmtL[cou],label = AllTri[cou],markersize='10')
            plt.errorbar(xpl, ypl ,np.asarray(brrTRI.sigmaCP),np.asarray(bllTRI.sigmaCP),fmt=fmtL[cou],label = AllTri[cou],markersize='10')

        plt.xlabel('closure phase RR',fontsize=15)
        plt.ylabel('closure phases LL',fontsize=15)
        plt.title(sour+', snr cut: '+str(snr_cut)+' '+add_text,fontsize=15)
        #x1,x2,y1,y2 = plt.axis()
        #plt.axis((-20,380,y1,y2))
        plt.legend()
        plt.axhline(y=.0,color='k',linestyle='--')
        plt.axvline(x=.0,color='k',linestyle='--')
        plt.grid()
        plt.tight_layout()
        
        if saveF==True:
            if snr_cut > 0.:
                nameF = 'CPRR_CPdiff_'+sour+'_snr_'+str(int(snr_cut))+'_'+add_text+'.pdf'
            else:
                nameF = 'CPRR_CPdiff_'+sour+'_'+add_text+'.pdf'
            plt.savefig(nameF)
        plt.show()        

def plt_all_triangles_datetime_CPdiff(brr,bll,sour,snr_cut=0.,saveF=False, add_text=''):
    fmtL = ['bo','ro','go','co','mo','ko','b^','r^','g^','c^','m^','k^','bv','rv','gv','cv','mv','kv']    
    brr = brr[brr['snr'] > snr_cut]; bll = bll[bll['snr'] >snr_cut]
    brr = brr[brr.source==sour]
    bll = bll[bll.source==sour]
    brr, bll = match_2_dataframes(brr, bll, 'triangle')
    if np.shape(brr)[0]>2.:
        AllTri = sorted(list(set(brr.triangle)))
        plt.figure(figsize=(10,10))
        
        for cou in range(len(AllTri)):
            brrTRI = brr[brr.triangle==AllTri[cou]]
            bllTRI = bll[bll.triangle==AllTri[cou]]
            plt.errorbar(list(brrTRI.datetime), phase_diff(brrTRI.cphase,bllTRI.cphase)-np.mod(brrTRI.cphase,360),np.sqrt(np.asarray(brrTRI.sigmaCP)**2 +np.asarray(bllTRI.sigmaCP)**2),fmt=fmtL[cou],label = AllTri[cou],markersize='10')
            
        plt.xlabel('time',fontsize=15)
        plt.ylabel('closure phases RR-LL difference',fontsize=15)
        plt.title(sour+', snr cut: '+str(snr_cut)+' '+add_text,fontsize=15)
        plt.legend()
        plt.axhline(y=.0,color='k',linestyle='--')
        plt.grid()
        plt.tight_layout()
        if saveF==True:
            if snr_cut > 0.:
                nameF = 'time_CPdiff_'+sour+'_snr_'+str(int(snr_cut))+'_'+add_text+'.pdf'
            else:
                nameF = 'time_CPdiff_'+sour+'_'+add_text+'.pdf'
            plt.savefig(nameF)
        plt.show()