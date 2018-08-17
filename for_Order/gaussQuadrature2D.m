clc;clear;
matlabpool local 12;
tic;
%--------------------------------------------------------------------------------1.µ¼Èë¾Ö²¿Ä£ÐÍ»ý·Öµã×ø±êºÍSÖµ
dizhi='D:\TYB\20180310\mÎÄ¼þ\';
local_file='local-750N.dat';  
fileID = fopen([dizhi,local_file]);

for ii=1:6     %localÄ£ÐÍ¹²ÓÐÈý¸öset,Ã¿¸öset°üÀ¨Ò»¸öcoordºÍÒ»¸öpeÖµ
   
   if ii==1
       local_zhi= textscan(fileID,'%f %f %f %f %f','headerlines',6);                  %¶ÁÈ¡localÄ£ÐÍ×ø±êÖµ,½ÏÃ»ÓÐsetÊ±£¬±êÌâÐÐ¶àÁËÒ»ÐÐ
       local_shun_coord1=cell2mat(local_zhi);
       number_local_shun1=length(local_shun_coord1(:,1));
       quan_local_shun_pe1=zeros(number_local_shun1,8);
   end
   
    if ii==2
       local_zhi= textscan(fileID,'%f %f %f %f %f','headerlines',6);                  %¶ÁÈ¡localÄ£ÐÍ×ø±êÖµ,½ÏÃ»ÓÐsetÊ±£¬±êÌâÐÐ¶àÁËÒ»ÐÐ
       local_shun_coord2=cell2mat(local_zhi);
       number_local_shun2=length(local_shun_coord2(:,1));
       quan_local_shun_pe2=zeros(number_local_shun2,8);
    end
   
   if ii==3
       local_zhi= textscan(fileID,'%f %f %f %f %f','headerlines',6);                  %¶ÁÈ¡localÄ£ÐÍÎÈÌ¬set×ø±êÖµ, ×¢ÒâÕâÀïµÄlocalÄ£ÐÍÖ»È¡Ò»Ìõ----Ö»È¡Ò»Ìõ---Ö»È¡Ò»Ìõ
       local_wen_coord=cell2mat(local_zhi);
       number_local_wen=length(local_wen_coord(:,1));
       quan_local_wen_pe=zeros(number_local_wen,8);
   end 
      
    if ii==4
       local_zhi= textscan(fileID,'%f %f %f %f %f %f %f %f','headerlines',6);                  %¶ÁÈ¡localÄ£ÐÍSÖµ
       local_shun_pe1=cell2mat(local_zhi);
       for mm=1:length(local_shun_pe1(:,1))
           hangshu=find(local_shun_coord1(:,1)==local_shun_pe1(mm,1));
           hangshu1=hangshu(1)+local_shun_pe1(mm,2)-1;
           quan_local_shun_pe1(hangshu1,3:8)=local_shun_pe1(mm,3:8);
       end
        quan_local_shun_pe1(:,1)=local_shun_coord1(:,1);
         quan_local_shun_pe1(:,2)=local_shun_coord1(:,2);
    end 

    if ii==5
       local_zhi= textscan(fileID,'%f %f %f %f %f %f %f %f','headerlines',6);                  %¶ÁÈ¡localÄ£ÐÍSÖµ
       local_shun_pe2=cell2mat(local_zhi);
       for mm=1:length(local_shun_pe2(:,1))
           hangshu=find(local_shun_coord2(:,1)==local_shun_pe2(mm,1));
           hangshu1=hangshu(1)+local_shun_pe2(mm,2)-1;
           quan_local_shun_pe2(hangshu1,3:8)=local_shun_pe2(mm,3:8);
       end
        quan_local_shun_pe2(:,1)=local_shun_coord2(:,1);
         quan_local_shun_pe2(:,2)=local_shun_coord2(:,2);
   end
   
    if ii==6
       local_zhi= textscan(fileID,'%f %f %f %f %f %f %f %f','headerlines',6);                  %¶ÁÈ¡localÄ£ÐÍSÖµ
       local_wen_pe=cell2mat(local_zhi);
       for mm=1:length(local_wen_pe(:,1))
           hangshu=find(local_wen_coord(:,1)==local_wen_pe(mm,1));
           hangshu1=hangshu(1)+local_wen_pe(mm,2)-1;
           quan_local_wen_pe(hangshu1,3:8)=local_wen_pe(mm,3:8);
       end
        quan_local_wen_pe(:,1)=local_wen_coord(:,1);
         quan_local_wen_pe(:,2)=local_wen_coord(:,2);
    end
   
end
   fclose(fileID);
 
%--------------------------------------------------------------------------------------------2.µ¼ÈëÈ«¾Ö×ø±êÖµ,·Ö¿éµÄ£¬Á½¸öË²Ì¬ºÍÖÐ¼äÎÈÌ¬ÇøÓò£¬Í¬Ê±½øÐÐ×ø±ê±ä»»


glob_file='global-750N.dat';  
fileID = fopen([dizhi,glob_file]);

for jj=1:3     %globalÄ£ÐÍ¹²ÓÐÈý¸öset,¼´Á½Ë²Ì¬ºÍÖÐ¼ä±í²ã£¬Ã¿¸ösetÖ»°üÀ¨Ò»¸öcoord
   if jj==1
       global_zhi= textscan(fileID,'%f %f %f %f %f','headerlines',6);                  %¶ÁÈ¡localÄ£ÐÍ×ø±êÖµ,½ÏÃ»ÓÐsetÊ±£¬±êÌâÐÐ¶àÁËÒ»ÐÐ
       global_shun_coord1=cell2mat(global_zhi);
   end
   
   if jj==2
       global_zhi= textscan(fileID,'%f %f %f %f %f','headerlines',6);                  %¶ÁÈ¡localÄ£ÐÍ×ø±êÖµ,½ÏÃ»ÓÐsetÊ±£¬±êÌâÐÐ¶àÁËÒ»ÐÐ
       global_shun_coord2=cell2mat(global_zhi);
   end
   
   if jj==3
       global_zhi= textscan(fileID,'%f %f %f %f %f','headerlines',6);                  %¶ÁÈ¡localÄ£ÐÍ×ø±êÖµ,½ÏÃ»ÓÐsetÊ±£¬±êÌâÐÐ¶àÁËÒ»ÐÐ
       global_wen_coord=cell2mat(global_zhi);
       %global_wen_coord=sortrows(global_wen_coord,[3,4]);
   end
   
end
fclose(fileID);
%--------------------------------------------------------------------------------------------3.È«¾Ö×ø±êÖµ×ø±ê±ä»»----------²»Í¬º¸·ìÐèÒª¸ÄµÄµØ·½
origin=[0,0,0];                                                     %ÐèÒªÃ¿Ìõº¸·ìµ¥¶ÀÈ·¶¨-----------------------²»Í¬º¸·ìÐèÒª¸ÄµÄµØ·½

x=[1,0,0];                                                            %¶ÔÓ¦È«¾Ö×ø±êÏµÉÏxÖáÉÏµÄµã-----------------------²»Í¬º¸·ìÐèÒª¸ÄµÄµØ·½
y=[0,1,0];                                                           %¶ÔÓ¦È«¾Ö×ø±êÏµÉÏyÖáÉÏµÄµã-----------------------²»Í¬º¸·ìÐèÒª¸ÄµÄµØ·½
z=[0,0,1];                                                          %¶ÔÓ¦È«¾Ö×ø±êÏµÉÏzÖáÉÏµÄµã-----------------------²»Í¬º¸·ìÐèÒª¸ÄµÄµØ·½ 
x_unit=(x-origin)./norm(x-origin);                                                                        %½«ÏòÁ¿µ¥Î»»¯
y_unit=(y-origin)./norm(y-origin);
z_unit=(z-origin)./norm(z-origin);
R_gl=[x_unit',y_unit',z_unit'];                                                                             %¾Ö²¿×ø±êµ¥Î»ÏòÁ¿ÔÚÈ«¾Ö×ø±êÏµÏÂµÄ×ø±ê£¬¼´·½ÏòÓàÏÒ¾ØÕó                                                                                        
R_lg=R_gl';                                                                                                        %×ªÖÃ¼´ÎªÄæ¾ØÕó£¬ÎªÈ«¾Ö×ø±êÔÚ¾Ö²¿×ø±êÏµÏÂµÄ×ø±ê

glob_zuobiao_shun1= global_shun_coord1(:,3:5);            
for kk=1:length(glob_zuobiao_shun1(:,1))
       glob_local_shun1(kk,3:5)=R_lg*glob_zuobiao_shun1(kk,:)'-R_lg*origin';    %£¡£¡Ö»ÓÐÈýÐÐ
       glob_local_shun1(kk,1:2)=global_shun_coord1(kk,1:2);
end

glob_zuobiao_shun2= global_shun_coord2(:,3:5);            
for kk=1:length(glob_zuobiao_shun2(:,1))
       glob_local_shun2(kk,3:5)=R_lg*glob_zuobiao_shun2(kk,:)'-R_lg*origin';
        glob_local_shun2(kk,1:2)=global_shun_coord2(kk,1:2);
end

glob_zuobiao_wen= global_wen_coord(:,3:5);            
for kk=1:length(glob_zuobiao_wen(:,1))
       glob_local_wen(kk,3:5)=R_lg*glob_zuobiao_wen(kk,:)'-R_lg*origin';
       glob_local_wen(kk,1:2)=global_wen_coord(kk,1:2);
end

G2_L2_juli=[0, 0, 0, -720, 0];                                %ºóÈý¸öÎª×ø±ê¡£¶ÔÓÚt2-¼´º¸½Ó½áÊøÇøÓò£¬globalÄ£ÐÍµÄÎ»ÖÃ¿ÉÄÜ²»Í¬ÓÚlocalÇøÓò£¬ÐèÒª×ø±êÑØÏàÓ¦µÄÏòÁ¿±ä»¯
for mm=1:length(glob_local_shun2(:,1))
glob_local_shun2(mm,:)=glob_local_shun2(mm,:)+G2_L2_juli;
end

%--------------------------------------------------------------------------------------------4.È«¾ÖË²Ì¬ÇøÓò»ý·ÖµãÕÒËùÔÚµÄµ¥Ôª£¬½«È«¾ÖµÄ»ý·ÖµãÓ³Éäµ½µÈ²Îµ¥ÔªÉÏ£¬ÔÙÍ¨¹ýµ¥ÔªÄÚ»ý·ÖµãµÄpeÖµ²åÖµ£¬Ð´Èë
 LL=[ -1,1,1;-1,-1,1;-1,1,-1;-1,-1,-1;1,1,1;1,-1,1;1,1,-1;1,-1,-1];
syms a b c
a0=LL(:,1)*a;
b0=LL(:,2)*b;
c0=LL(:,3)*c;
N=1/8*(1+a0).*(1+b0).*(1+c0);
S0=[N(1),0,0,N(2),0,0,N(3),0,0,N(4),0,0,N(5),0,0,N(6),0,0,N(7),0,0,N(8),0,0;
    0,N(1),0,0,N(2),0,0,N(3),0,0,N(4),0,0,N(5),0,0,N(6),0,0,N(7),0,0,N(8),0;
    0,0,N(1), 0,0,N(2), 0,0,N(3), 0,0,N(4), 0,0,N(5), 0,0,N(6), 0,0,N(7), 0,0,N(8)];     %µÈ²Îµ¥Ôª¶ÔÓ¦µÄ×ø±êÇóÈ¡

S=[N(1),0,0,0,0,0,N(2),0,0,0,0,0,N(3),0,0,0,0,0,N(4),0,0,0,0,0,N(5),0,0,0,0,0,N(6),0,0,0,0,0,N(7),0,0,0,0,0,N(8),0,0,0,0,0;
    0,N(1),0,0,0,0,0,N(2),0,0,0,0,0,N(3),0,0,0,0,0,N(4),0,0,0,0,0,N(5),0,0,0,0,0,N(6),0,0,0,0,0,N(7),0,0,0,0,0,N(8),0,0,0,0;
    0,0,N(1),0,0,0,0,0,N(2),0,0,0,0,0,N(3),0,0,0,0,0,N(4),0,0,0,0,0,N(5),0,0,0,0,0,N(6),0,0,0, 0,0,N(7),0,0,0,0,0,N(8),0,0,0;
    0,0,0,N(1),0,0,0,0,0,N(2),0,0,0,0,0,N(3),0,0,0,0,0,N(4),0,0,0,0,0,N(5),0,0,0,0,0,N(6),0,0, 0,0,0,N(7),0,0,0,0,0,N(8),0,0;
    0,0,0,0,N(1),0,0,0,0,0,N(2),0,0,0,0,0,N(3),0,0,0,0,0,N(4),0,0,0,0,0,N(5),0,0,0,0,0,N(6),0, 0,0,0,0,N(7),0,0,0,0,0,N(8),0;
    0,0,0,0,0,N(1),0,0,0,0,0,N(2),0,0,0,0,0,N(3),0,0,0,0,0,N(4),0,0,0,0,0,N(5),0,0,0,0,0,N(6), 0,0,0,0,0,N(7),0,0,0,0,0,N(8);];      %6*6µÄ¾ØÕó£¬²åÖµËùÐèµÄS¾ØÕó

%--------------------------------------------------------------------------------------------4.1 %Ë²Ì¬µÚÒ»¿éÕÒ×î½üµã

parfor i=1:length(global_shun_coord1(:,1))                 %k¿ÉÒÔ¸Ä³Éparfor, global_shun_coord1Ò²Ó¦¸Ã¸Ä³É×ø±ê±ä»»ºóµÄglob_local_shun1
    d=sqrt(( local_shun_coord1(:,3)-glob_local_shun1(i,3)).^2 + ( local_shun_coord1(:,4)-glob_local_shun1(i,4)).^2+( local_shun_coord1(:,5)-glob_local_shun1(i,5)).^2);    %È«¾Ö×ø±êÓ¦Îª×ø±ê±ä»»ºóµÄ×ø±ê 
       JL_data=[d];
       [u,v]=sort(JL_data);        %ÐèÒªv(1)----´ú±íÊÇÎÄ¼þÖÐµÄµÚ¼¸¸ö¾Ö²¿µ¥Ôª»ý·Öµã£¬½ø¶ø¿ÉÒÔµÃµ½¸Ãµ¥Ôª»ý·Öµã×ø±ê£»u(1)´ú±í×îÐ¡¾àÀëÖµÎª¶àÉÙ           
       eight_ip=(ceil(v(1)/8)-1)*8+[1:8];          %¶ÔÓ¦µÄ¾Ö²¿µ¥ÔªµÄ8¸ö»ý·Öµã±àºÅ
       %v(1)    
      GL0=[local_shun_coord1(eight_ip(1),3:5)'; local_shun_coord1(eight_ip(2),3:5)'; local_shun_coord1(eight_ip(3),3:5)'; local_shun_coord1(eight_ip(4),3:5)'; local_shun_coord1(eight_ip(5),3:5)'; local_shun_coord1(eight_ip(6),3:5)'; local_shun_coord1(eight_ip(7),3:5)'; local_shun_coord1(eight_ip(8),3:5)'];  %¾Ö²¿×ø±ê8¸ö»ý·ÖµãµÄ×ø±ê
      global_local_int=S0*GL0;                  %»ñµÃ¾Ö²¿×ø±êÏµÏÂ²åÖµµÄ×ø±ê
      B=solve(global_local_int(1)==glob_local_shun1(i,3),global_local_int(2)==glob_local_shun1(i,4),global_local_int(3)==glob_local_shun1(i,5),a,b,c);              %»ñµÃµÈ²Îµ¥ÔªÖÐ¶ÔÓ¦µÄ×ø±ê¡£
      B1=vpa(B.a);%vpa¿ØÖÆ¾«¶È
      zz=subs(B1,1000);%ÀûÓÃ1000´úÌæB1ÖÐµÄÄ¬ÈÏ·ûºÅ
      zzz=abs(zz)-1;%Ä£»òÕß¾ø¶ÔÖµ¼ÆËã
      [zhi,wei]=min(abs(zzz));
      B1=real(B1(wei));%Êµ²¿¼ÆËã
      B1=double(B1);
      qqq1(i)=B1;
      
      B2=vpa(B.b);
      zz=subs(B2,1000);
      zzz=abs(zz)-1;
      [zhi,wei]=min(abs(zzz));
      B2=real(B2(wei));
      B2=double(B2);
      qqq2(i)=B2;
      
      B3=vpa(B.c);
      zz=subs(B3,1000)
      zzz=abs(zz)-1;
      [zhi,wei]=min(abs(zzz));
      B3=real(B3(wei));
      B3=double(B3);
      qqq3(i)=B3;      
      
     GL=[quan_local_shun_pe1(eight_ip(1),3:8)';quan_local_shun_pe1(eight_ip(2),3:8)';quan_local_shun_pe1(eight_ip(3),3:8)';quan_local_shun_pe1(eight_ip(4),3:8)';quan_local_shun_pe1(eight_ip(5),3:8)';quan_local_shun_pe1(eight_ip(6),3:8)';quan_local_shun_pe1(eight_ip(7),3:8)';quan_local_shun_pe1(eight_ip(8),3:8)'];  %¾Ö²¿×ø±ê8¸ö»ý·ÖµãµÄpe·ÖÁ¿
     PL=S*GL;                                                                   
     A_gl_t1(i,:)=subs(PL,{a,b,c},{B1,B2,B3})';
end

parfor i=1:length(global_shun_coord2(:,1))                 %k¿ÉÒÔ¸Ä³Éparfor
    i
    d=sqrt(( local_shun_coord2(:,3)-glob_local_shun2(i,3)).^2 + ( local_shun_coord2(:,4)-glob_local_shun2(i,4)).^2+( local_shun_coord2(:,5)-glob_local_shun2(i,5)).^2);    %È«¾Ö×ø±êÓ¦Îª×ø±ê±ä»»ºóµÄ×ø±ê 
       JL_data=[d];
       [u,v]=sort(JL_data);        %ÐèÒªv(1)----´ú±íÊÇÎÄ¼þÖÐµÄµÚ¼¸¸ö¾Ö²¿µ¥Ôª»ý·Öµã£¬½ø¶ø¿ÉÒÔµÃµ½¸Ãµ¥Ôª»ý·Öµã×ø±ê£»u(1)´ú±í×îÐ¡¾àÀëÖµÎª¶àÉÙ           
       eight_ip=(ceil(v(1)/8)-1)*8+[1:8];          %¶ÔÓ¦µÄ¾Ö²¿µ¥ÔªµÄ8¸ö»ý·Öµã±àºÅ
       
       %v(1)
      GL0=[ local_shun_coord2(eight_ip(1),3:5)'; local_shun_coord2(eight_ip(2),3:5)'; local_shun_coord2(eight_ip(3),3:5)'; local_shun_coord2(eight_ip(4),3:5)'; local_shun_coord2(eight_ip(5),3:5)'; local_shun_coord2(eight_ip(6),3:5)'; local_shun_coord2(eight_ip(7),3:5)'; local_shun_coord2(eight_ip(8),3:5)'];  %¾Ö²¿×ø±ê8¸ö»ý·ÖµãµÄ×ø±ê
      global_local_int=S0*GL0;                  %»ñµÃ¾Ö²¿×ø±êÏµÏÂ²åÖµµÄ×ø±ê
      B=solve(global_local_int(1)-glob_local_shun2(i,3),global_local_int(2)-glob_local_shun2(i,4),global_local_int(3)-glob_local_shun2(i,5),a,b,c);              %»ñµÃµÈ²Îµ¥ÔªÖÐ¶ÔÓ¦µÄ×ø±ê¡£'PrincipalValue',true,
      B1=vpa(B.a);
      zz=subs(B1,1000);
      zzz=abs(zz)-1;
      [zhi,wei]=min(abs(zzz));
      B1=real(B1(wei));
      B1=double(B1);
      sss1(i)=B1;
      
      B2=vpa(B.b);
      zz=subs(B2,1000);
      zzz=abs(zz)-1;
      [zhi,wei]=min(abs(zzz));
      B2=real(B2(wei));
      B2=double(B2);
      sss2(i)=B2;
      
      B3=vpa(B.c);
      zz=subs(B3,1000)
      zzz=abs(zz)-1;
      [zhi,wei]=min(abs(zzz));
      B3=real(B3(wei));
      B3=double(B3);
      sss3(i)=B3;
      
     GL=[quan_local_shun_pe2(eight_ip(1),3:8)';quan_local_shun_pe2(eight_ip(2),3:8)';quan_local_shun_pe2(eight_ip(3),3:8)';quan_local_shun_pe2(eight_ip(4),3:8)';quan_local_shun_pe2(eight_ip(5),3:8)';quan_local_shun_pe2(eight_ip(6),3:8)';quan_local_shun_pe2(eight_ip(7),3:8)';quan_local_shun_pe2(eight_ip(8),3:8)'];  %¾Ö²¿×ø±ê8¸ö»ý·ÖµãµÄpe·ÖÁ¿
     PL=S*GL;                                                                   
     A_gl_t2(i,:)=subs(PL,{a,b,c},{B1,B2,B3})';
end

parfor i=1:length(glob_local_wen(:,1))                 %k¿ÉÒÔ¸Ä³Éparfor
    i
    d=sqrt((local_wen_coord(:,3)- glob_local_wen(i,3)).^2 + (local_wen_coord(:,4)- glob_local_wen(i,4)).^2+(local_wen_coord(:,5)- glob_local_wen(i,5)).^2);    %È«¾Ö×ø±êÓ¦Îª×ø±ê±ä»»ºóµÄ×ø±ê 
       JL_data=[d];
       [u,v]=sort(JL_data);        %ÐèÒªv(1)----´ú±íÊÇÎÄ¼þÖÐµÄµÚ¼¸¸ö¾Ö²¿µ¥Ôª»ý·Öµã£¬½ø¶ø¿ÉÒÔµÃµ½¸Ãµ¥Ôª»ý·Öµã×ø±ê£»u(1)´ú±í×îÐ¡¾àÀëÖµÎª¶àÉÙ           
       eight_ip=(ceil(v(1)/8)-1)*8+[1:8];          %¶ÔÓ¦µÄ¾Ö²¿µ¥ÔªµÄ8¸ö»ý·Öµã±àºÅ
       v(1);    
      GL0=[local_wen_coord(eight_ip(1),3:5)'; local_wen_coord(eight_ip(2),3:5)';local_wen_coord(eight_ip(3),3:5)'; local_wen_coord(eight_ip(4),3:5)'; local_wen_coord(eight_ip(5),3:5)'; local_wen_coord(eight_ip(6),3:5)'; local_wen_coord(eight_ip(7),3:5)'; local_wen_coord(eight_ip(8),3:5)'];  %¾Ö²¿×ø±ê8¸ö»ý·ÖµãµÄ×ø±ê
      global_local_int=S0*GL0;                  %»ñµÃ¾Ö²¿×ø±êÏµÏÂ²åÖµµÄ×ø±ê
      B=solve(global_local_int(1)==glob_local_wen(i,3),global_local_int(2)==local_wen_coord(v(1),4),global_local_int(3)==glob_local_wen(i,5),a,b,c);              %------×¢ÒâÕâ¸ö²»Ò»Ñù,Y×ø±êÓÃµÄ¾Ö²¿µ¥ÔªµÄ»ñµÃµÈ²Îµ¥ÔªÖÐ¶ÔÓ¦µÄ×ø±ê¡£
      B1=vpa(B.a);
      zz=subs(B1,1000);
      zzz=abs(zz)-1;
      [zhi,wei]=min(abs(zzz));
      B1=real(B1(wei));
      B1=double(B1);
      rrr1(i)=B1;
      
      B2=vpa(B.b);
      zz=subs(B2,1000);
      zzz=abs(zz)-1;
      [zhi,wei]=min(abs(zzz));
      B2=real(B2(wei));
      B2=double(B2);
      rrr2(i)=B2;
      
      B3=vpa(B.c);
      zz=subs(B3,1000)
      zzz=abs(zz)-1;
      [zhi,wei]=min(abs(zzz));
      B3=real(B3(wei));
      B3=double(B3);
      rrr3(i)=B3;
      
     GL=[quan_local_wen_pe(eight_ip(1),3:8)';quan_local_wen_pe(eight_ip(2),3:8)';quan_local_wen_pe(eight_ip(3),3:8)';quan_local_wen_pe(eight_ip(4),3:8)';quan_local_wen_pe(eight_ip(5),3:8)';quan_local_wen_pe(eight_ip(6),3:8)';quan_local_wen_pe(eight_ip(7),3:8)';quan_local_wen_pe(eight_ip(8),3:8)'];  %¾Ö²¿×ø±ê8¸ö»ý·ÖµãµÄpe·ÖÁ¿
     PL=S*GL;                                                                   
     A_gl_s(i,:)=subs(PL,{a,b,c},{B1,B2,B3})';
end


%------------------5.È«¾ÖÄ£ÐÍ8¸öµ¥Ôª»ý·ÖµãµÄÓ¦±ä·ÖÁ¿²åÖµ³öÖÐÐÄµãµÄÓ¦±ä·ÖÁ¿,½øÐÐ×ø±ê±ä»»ºóÐ´ÈëÎÄ±¾ÖÐ
ddd='elset_p750N_40mm.txt';
fid_elset_p1=fopen([dizhi,ddd],'w');
format_elset_p1= '*Elset, elset=Set-gf-%d, internal, instance=PART-1-1 \r\n %d \r\n';

eee='S_p750N_40mm.txt';
fid_pe_p1=fopen([dizhi,eee],'w');
format_pe_p1 = '*Initial Conditions, type=Stress \r\nSet-gf-%d, %d, %d, %d, %d, %d, %d\r\n';

%---------------------------------------------5.1È«¾ÖÄ£ÐÍº¸½ÓÆðÊ¼ÇøÓòÐ´Èë
for j=1:length(glob_local_shun1)/8
    aaa=glob_local_shun1((j-1)*8+1,1)
    fprintf(fid_elset_p1,format_elset_p1,j,aaa);
    
    num=(1:8)+(j-1)*8;                      %¶ÔÓ¦µÄ8¸ö»ý·Öµã±àºÅ
    C=[A_gl_t1(num(1),:)';A_gl_t1(num(2),:)';A_gl_t1(num(3),:)';A_gl_t1(num(4),:)';A_gl_t1(num(5),:)';A_gl_t1(num(6),:)';A_gl_t1(num(7),:)';A_gl_t1(num(8),:)'];        %¶ÔÓ¦µÄ8¸ö»ý·ÖµãPEÖµ
    CZ=S*C;                                       
    D(j,:)=subs(CZ,{a,b,c},{0,0,0});
    ebsino_L=[D(j,1),D(j,4),D(j,5);D(j,4),D(j,2),D(j,6);D(j,5),D(j,6),D(j,3)];
   ebsino_G=R_gl*ebsino_L*R_gl';
   DDD(j,:)=[ebsino_G(1,1),ebsino_G(2,2),ebsino_G(3,3),ebsino_G(1,2),ebsino_G(1,3),ebsino_G(2,3)];
    fprintf(fid_pe_p1,format_pe_p1,j,DDD(j,:));
   
end
%---------------------------------------------5.2È«¾ÖÄ£ÐÍº¸½Ó½áÊøÇøÓòÐ´Èë
for k=1:length(glob_local_shun2)/8
    aaa=glob_local_shun2((k-1)*8+1,1)
    fprintf(fid_elset_p1,format_elset_p1,k+j,aaa);
    
    num=(1:8)+(k-1)*8;                      %¶ÔÓ¦µÄ8¸ö»ý·Öµã±àºÅ
    C=[A_gl_t2(num(1),:)';A_gl_t2(num(2),:)';A_gl_t2(num(3),:)';A_gl_t2(num(4),:)';A_gl_t2(num(5),:)';A_gl_t2(num(6),:)';A_gl_t1(num(7),:)';A_gl_t2(num(8),:)'];        %¶ÔÓ¦µÄ8¸ö»ý·ÖµãPEÖµ
    CZ=S*C;                                       
    D(k,:)=subs(CZ,{a,b,c},{0,0,0}); 
   ebsino_L=[D(k,1),D(j,4),D(k,5);D(k,4),D(k,2),D(k,6);D(k,5),D(k,6),D(k,3)];
   ebsino_G=R_gl*ebsino_L*R_gl';
   DDD(k,:)=[ebsino_G(1,1),ebsino_G(2,2),ebsino_G(3,3),ebsino_G(1,2),ebsino_G(1,3),ebsino_G(2,3)];
    fprintf(fid_pe_p1,format_pe_p1,k+j,DDD(k,:));
end
%---------------------------------------------5.3È«¾ÖÄ£ÐÍº¸½ÓÎÈÌ¬ÇøÓòÐ´Èë

for jk=1:length(glob_local_wen)/8
     aaa=glob_local_wen((jk-1)*8+1,1)
    fprintf(fid_elset_p1,format_elset_p1,jk+j+k,aaa);
    
    num=(1:8)+(jk-1)*8;                      %¶ÔÓ¦µÄ8¸ö»ý·Öµã±àºÅ
    C=[A_gl_s(num(1),:)';A_gl_s(num(2),:)';A_gl_s(num(3),:)';A_gl_s(num(4),:)';A_gl_s(num(5),:)';A_gl_s(num(6),:)';A_gl_s(num(7),:)';A_gl_s(num(8),:)'];        %¶ÔÓ¦µÄ8¸ö»ý·ÖµãPEÖµ
    CZ=S*C;                                       
    D(jk,:)=subs(CZ,{a,b,c},{0,0,0}); 
   ebsino_L=[D(jk,1),D(jk,4),D(jk,5);D(jk,4),D(jk,2),D(jk,6);D(jk,5),D(jk,6),D(jk,3)];
   ebsino_G=R_gl*ebsino_L*R_gl';
   DDD(jk,:)=[ebsino_G(1,1),ebsino_G(2,2),ebsino_G(3,3),ebsino_G(1,2),ebsino_G(1,3),ebsino_G(2,3)];
    fprintf(fid_pe_p1,format_pe_p1,jk+j+k,DDD(jk,:));
end

fclose(fid_elset_p1);
fclose(fid_pe_p1);
matlabpool close;
toc;
 


