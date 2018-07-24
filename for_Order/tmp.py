clc;clear;
matlabpool local 12;
tic;
%--------------------------------------------------------------------------------
% 1.导入局部模型积分点坐标和S值
dizhi='D:\TYB\20180310\m文件\';
local_file='local-750N.dat';
fileID = fopen([dizhi,local_file]);

for ii=1:6     %local模型共有三个set,每个set包括一个coord和一个pe值

   if ii==1
       local_zhi= textscan(fileID,'%f %f %f %f %f','headerlines',6);                  %读取local模型坐标值,较没有set时，标题行多了一行
       local_shun_coord1=cell2mat(local_zhi);
       number_local_shun1=length(local_shun_coord1(:,1));
       quan_local_shun_pe1=zeros(number_local_shun1,8);
   end

    if ii==2
       local_zhi= textscan(fileID,'%f %f %f %f %f','headerlines',6);                  %读取local模型坐标值,较没有set时，标题行多了一行
       local_shun_coord2=cell2mat(local_zhi);
       number_local_shun2=length(local_shun_coord2(:,1));
       quan_local_shun_pe2=zeros(number_local_shun2,8);
    end

   if ii==3
       local_zhi= textscan(fileID,'%f %f %f %f %f','headerlines',6);                  %读取local模型稳态set坐标值, 注意这里的local模型只取一条----只取一条---只取一条
       local_wen_coord=cell2mat(local_zhi);
       number_local_wen=length(local_wen_coord(:,1));
       quan_local_wen_pe=zeros(number_local_wen,8);
   end

    if ii==4
       local_zhi= textscan(fileID,'%f %f %f %f %f %f %f %f','headerlines',6);                  %读取local模型S值
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
       local_zhi= textscan(fileID,'%f %f %f %f %f %f %f %f','headerlines',6);                  %读取local模型S值
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
       local_zhi= textscan(fileID,'%f %f %f %f %f %f %f %f','headerlines',6);                  %读取local模型S值
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

%--------------------------------------------------------------------------------------------2.导入全局坐标值,分块的，两个瞬态和中间稳态区域，同时进行坐标变换


glob_file='global-750N.dat';
fileID = fopen([dizhi,glob_file]);

for jj=1:3     %global模型共有三个set,即两瞬态和中间表层，每个set只包括一个coord
   if jj==1
       global_zhi= textscan(fileID,'%f %f %f %f %f','headerlines',6);                  %读取local模型坐标值,较没有set时，标题行多了一行
       global_shun_coord1=cell2mat(global_zhi);
   end

   if jj==2
       global_zhi= textscan(fileID,'%f %f %f %f %f','headerlines',6);                  %读取local模型坐标值,较没有set时，标题行多了一行
       global_shun_coord2=cell2mat(global_zhi);
   end

   if jj==3
       global_zhi= textscan(fileID,'%f %f %f %f %f','headerlines',6);                  %读取local模型坐标值,较没有set时，标题行多了一行
       global_wen_coord=cell2mat(global_zhi);
       %global_wen_coord=sortrows(global_wen_coord,[3,4]);
   end

end
fclose(fileID);
%--------------------------------------------------------------------------------------------3.全局坐标值坐标变换----------不同焊缝需要改的地方
origin=[0,0,0];                                                     %需要每条焊缝单独确定-----------------------不同焊缝需要改的地方

x=[1,0,0];                                                            %对应全局坐标系上x轴上的点-----------------------不同焊缝需要改的地方
y=[0,1,0];                                                           %对应全局坐标系上y轴上的点-----------------------不同焊缝需要改的地方
z=[0,0,1];                                                          %对应全局坐标系上z轴上的点-----------------------不同焊缝需要改的地方
x_unit=(x-origin)./norm(x-origin);                                                                        %将向量单位化
y_unit=(y-origin)./norm(y-origin);
z_unit=(z-origin)./norm(z-origin);
R_gl=[x_unit',y_unit',z_unit'];                                                                             %局部坐标单位向量在全局坐标系下的坐标，即方向余弦矩阵
R_lg=R_gl';                                                                                                        %转置即为逆矩阵，为全局坐标在局部坐标系下的坐标

glob_zuobiao_shun1= global_shun_coord1(:,3:5);
for kk=1:length(glob_zuobiao_shun1(:,1))
       glob_local_shun1(kk,3:5)=R_lg*glob_zuobiao_shun1(kk,:)'-R_lg*origin';    %！！只有三行
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

G2_L2_juli=[0, 0, 0, -720, 0];                                %后三个为坐标。对于t2-即焊接结束区域，global模型的位置可能不同于local区域，需要坐标沿相应的向量变化
for mm=1:length(glob_local_shun2(:,1))
glob_local_shun2(mm,:)=glob_local_shun2(mm,:)+G2_L2_juli;
end

%--------------------------------------------------------------------------------------------4.全局瞬态区域积分点找所在的单元，将全局的积分点映射到等参单元上，再通过单元内积分点的pe值插值，写入
 LL=[ -1,1,1;-1,-1,1;-1,1,-1;-1,-1,-1;1,1,1;1,-1,1;1,1,-1;1,-1,-1];
syms a b c
a0=LL(:,1)*a;
b0=LL(:,2)*b;
c0=LL(:,3)*c;
N=1/8*(1+a0).*(1+b0).*(1+c0);
S0=[N(1),0,0,N(2),0,0,N(3),0,0,N(4),0,0,N(5),0,0,N(6),0,0,N(7),0,0,N(8),0,0;
    0,N(1),0,0,N(2),0,0,N(3),0,0,N(4),0,0,N(5),0,0,N(6),0,0,N(7),0,0,N(8),0;
    0,0,N(1), 0,0,N(2), 0,0,N(3), 0,0,N(4), 0,0,N(5), 0,0,N(6), 0,0,N(7), 0,0,N(8)];     %等参单元对应的坐标求取

S=[N(1),0,0,0,0,0,N(2),0,0,0,0,0,N(3),0,0,0,0,0,N(4),0,0,0,0,0,N(5),0,0,0,0,0,N(6),0,0,0,0,0,N(7),0,0,0,0,0,N(8),0,0,0,0,0;
    0,N(1),0,0,0,0,0,N(2),0,0,0,0,0,N(3),0,0,0,0,0,N(4),0,0,0,0,0,N(5),0,0,0,0,0,N(6),0,0,0,0,0,N(7),0,0,0,0,0,N(8),0,0,0,0;
    0,0,N(1),0,0,0,0,0,N(2),0,0,0,0,0,N(3),0,0,0,0,0,N(4),0,0,0,0,0,N(5),0,0,0,0,0,N(6),0,0,0, 0,0,N(7),0,0,0,0,0,N(8),0,0,0;
    0,0,0,N(1),0,0,0,0,0,N(2),0,0,0,0,0,N(3),0,0,0,0,0,N(4),0,0,0,0,0,N(5),0,0,0,0,0,N(6),0,0, 0,0,0,N(7),0,0,0,0,0,N(8),0,0;
    0,0,0,0,N(1),0,0,0,0,0,N(2),0,0,0,0,0,N(3),0,0,0,0,0,N(4),0,0,0,0,0,N(5),0,0,0,0,0,N(6),0, 0,0,0,0,N(7),0,0,0,0,0,N(8),0;
    0,0,0,0,0,N(1),0,0,0,0,0,N(2),0,0,0,0,0,N(3),0,0,0,0,0,N(4),0,0,0,0,0,N(5),0,0,0,0,0,N(6), 0,0,0,0,0,N(7),0,0,0,0,0,N(8);];      %6*6的矩阵，插值所需的S矩阵

%--------------------------------------------------------------------------------------------4.1 %瞬态第一块找最近点

parfor i=1:length(global_shun_coord1(:,1))                 %k可以改成parfor, global_shun_coord1也应该改成坐标变换后的glob_local_shun1
    d=sqrt(( local_shun_coord1(:,3)-glob_local_shun1(i,3)).^2 + ( local_shun_coord1(:,4)-glob_local_shun1(i,4)).^2+( local_shun_coord1(:,5)-glob_local_shun1(i,5)).^2);    %全局坐标应为坐标变换后的坐标
       JL_data=[d];
       [u,v]=sort(JL_data);        %需要v(1)----代表是文件中的第几个局部单元积分点，进而可以得到该单元积分点坐标；u(1)代表最小距离值为多少
       eight_ip=(ceil(v(1)/8)-1)*8+[1:8];          %对应的局部单元的8个积分点编号
       %v(1)
      GL0=[local_shun_coord1(eight_ip(1),3:5)'; local_shun_coord1(eight_ip(2),3:5)'; local_shun_coord1(eight_ip(3),3:5)'; local_shun_coord1(eight_ip(4),3:5)'; local_shun_coord1(eight_ip(5),3:5)'; local_shun_coord1(eight_ip(6),3:5)'; local_shun_coord1(eight_ip(7),3:5)'; local_shun_coord1(eight_ip(8),3:5)'];  %局部坐标8个积分点的坐标
      global_local_int=S0*GL0;                  %获得局部坐标系下插值的坐标
      B=solve(global_local_int(1)==glob_local_shun1(i,3),global_local_int(2)==glob_local_shun1(i,4),global_local_int(3)==glob_local_shun1(i,5),a,b,c);              %获得等参单元中对应的坐标。
      B1=vpa(B.a);%vpa控制精度
      zz=subs(B1,1000);%利用1000代替B1中的默认符号
      zzz=abs(zz)-1;%模或者绝对值计算
      [zhi,wei]=min(abs(zzz));
      B1=real(B1(wei));%实部计算
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

     GL=[quan_local_shun_pe1(eight_ip(1),3:8)';quan_local_shun_pe1(eight_ip(2),3:8)';quan_local_shun_pe1(eight_ip(3),3:8)';quan_local_shun_pe1(eight_ip(4),3:8)';quan_local_shun_pe1(eight_ip(5),3:8)';quan_local_shun_pe1(eight_ip(6),3:8)';quan_local_shun_pe1(eight_ip(7),3:8)';quan_local_shun_pe1(eight_ip(8),3:8)'];  %局部坐标8个积分点的pe分量
     PL=S*GL;
     A_gl_t1(i,:)=subs(PL,{a,b,c},{B1,B2,B3})';
end

parfor i=1:length(global_shun_coord2(:,1))                 %k可以改成parfor
    i
    d=sqrt(( local_shun_coord2(:,3)-glob_local_shun2(i,3)).^2 + ( local_shun_coord2(:,4)-glob_local_shun2(i,4)).^2+( local_shun_coord2(:,5)-glob_local_shun2(i,5)).^2);    %全局坐标应为坐标变换后的坐标
       JL_data=[d];
       [u,v]=sort(JL_data);        %需要v(1)----代表是文件中的第几个局部单元积分点，进而可以得到该单元积分点坐标；u(1)代表最小距离值为多少
       eight_ip=(ceil(v(1)/8)-1)*8+[1:8];          %对应的局部单元的8个积分点编号

       %v(1)
      GL0=[ local_shun_coord2(eight_ip(1),3:5)'; local_shun_coord2(eight_ip(2),3:5)'; local_shun_coord2(eight_ip(3),3:5)'; local_shun_coord2(eight_ip(4),3:5)'; local_shun_coord2(eight_ip(5),3:5)'; local_shun_coord2(eight_ip(6),3:5)'; local_shun_coord2(eight_ip(7),3:5)'; local_shun_coord2(eight_ip(8),3:5)'];  %局部坐标8个积分点的坐标
      global_local_int=S0*GL0;                  %获得局部坐标系下插值的坐标
      B=solve(global_local_int(1)-glob_local_shun2(i,3),global_local_int(2)-glob_local_shun2(i,4),global_local_int(3)-glob_local_shun2(i,5),a,b,c);              %获得等参单元中对应的坐标。'PrincipalValue',true,
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

     GL=[quan_local_shun_pe2(eight_ip(1),3:8)';quan_local_shun_pe2(eight_ip(2),3:8)';quan_local_shun_pe2(eight_ip(3),3:8)';quan_local_shun_pe2(eight_ip(4),3:8)';quan_local_shun_pe2(eight_ip(5),3:8)';quan_local_shun_pe2(eight_ip(6),3:8)';quan_local_shun_pe2(eight_ip(7),3:8)';quan_local_shun_pe2(eight_ip(8),3:8)'];  %局部坐标8个积分点的pe分量
     PL=S*GL;
     A_gl_t2(i,:)=subs(PL,{a,b,c},{B1,B2,B3})';
end

parfor i=1:length(glob_local_wen(:,1))                 %k可以改成parfor
    i
    d=sqrt((local_wen_coord(:,3)- glob_local_wen(i,3)).^2 + (local_wen_coord(:,4)- glob_local_wen(i,4)).^2+(local_wen_coord(:,5)- glob_local_wen(i,5)).^2);    %全局坐标应为坐标变换后的坐标
       JL_data=[d];
       [u,v]=sort(JL_data);        %需要v(1)----代表是文件中的第几个局部单元积分点，进而可以得到该单元积分点坐标；u(1)代表最小距离值为多少
       eight_ip=(ceil(v(1)/8)-1)*8+[1:8];          %对应的局部单元的8个积分点编号
       v(1);
      GL0=[local_wen_coord(eight_ip(1),3:5)'; local_wen_coord(eight_ip(2),3:5)';local_wen_coord(eight_ip(3),3:5)'; local_wen_coord(eight_ip(4),3:5)'; local_wen_coord(eight_ip(5),3:5)'; local_wen_coord(eight_ip(6),3:5)'; local_wen_coord(eight_ip(7),3:5)'; local_wen_coord(eight_ip(8),3:5)'];  %局部坐标8个积分点的坐标
      global_local_int=S0*GL0;                  %获得局部坐标系下插值的坐标
      B=solve(global_local_int(1)==glob_local_wen(i,3),global_local_int(2)==local_wen_coord(v(1),4),global_local_int(3)==glob_local_wen(i,5),a,b,c);              %------注意这个不一样,Y坐标用的局部单元的获得等参单元中对应的坐标。
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

     GL=[quan_local_wen_pe(eight_ip(1),3:8)';quan_local_wen_pe(eight_ip(2),3:8)';quan_local_wen_pe(eight_ip(3),3:8)';quan_local_wen_pe(eight_ip(4),3:8)';quan_local_wen_pe(eight_ip(5),3:8)';quan_local_wen_pe(eight_ip(6),3:8)';quan_local_wen_pe(eight_ip(7),3:8)';quan_local_wen_pe(eight_ip(8),3:8)'];  %局部坐标8个积分点的pe分量
     PL=S*GL;
     A_gl_s(i,:)=subs(PL,{a,b,c},{B1,B2,B3})';
end


%------------------5.全局模型8个单元积分点的应变分量插值出中心点的应变分量,进行坐标变换后写入文本中
ddd='elset_p750N_40mm.txt';
fid_elset_p1=fopen([dizhi,ddd],'w');
format_elset_p1= '*Elset, elset=Set-gf-%d, internal, instance=PART-1-1 \r\n %d \r\n';

eee='S_p750N_40mm.txt';
fid_pe_p1=fopen([dizhi,eee],'w');
format_pe_p1 = '*Initial Conditions, type=Stress \r\nSet-gf-%d, %d, %d, %d, %d, %d, %d\r\n';

%---------------------------------------------5.1全局模型焊接起始区域写入
for j=1:length(glob_local_shun1)/8
    aaa=glob_local_shun1((j-1)*8+1,1)
    fprintf(fid_elset_p1,format_elset_p1,j,aaa);

    num=(1:8)+(j-1)*8;                      %对应的8个积分点编号
    C=[A_gl_t1(num(1),:)';A_gl_t1(num(2),:)';A_gl_t1(num(3),:)';A_gl_t1(num(4),:)';A_gl_t1(num(5),:)';A_gl_t1(num(6),:)';A_gl_t1(num(7),:)';A_gl_t1(num(8),:)'];        %对应的8个积分点PE值
    CZ=S*C;
    D(j,:)=subs(CZ,{a,b,c},{0,0,0});
    ebsino_L=[D(j,1),D(j,4),D(j,5);D(j,4),D(j,2),D(j,6);D(j,5),D(j,6),D(j,3)];
   ebsino_G=R_gl*ebsino_L*R_gl';
   DDD(j,:)=[ebsino_G(1,1),ebsino_G(2,2),ebsino_G(3,3),ebsino_G(1,2),ebsino_G(1,3),ebsino_G(2,3)];
    fprintf(fid_pe_p1,format_pe_p1,j,DDD(j,:));

end
%---------------------------------------------5.2全局模型焊接结束区域写入
for k=1:length(glob_local_shun2)/8
    aaa=glob_local_shun2((k-1)*8+1,1)
    fprintf(fid_elset_p1,format_elset_p1,k+j,aaa);

    num=(1:8)+(k-1)*8;                      %对应的8个积分点编号
    C=[A_gl_t2(num(1),:)';A_gl_t2(num(2),:)';A_gl_t2(num(3),:)';A_gl_t2(num(4),:)';A_gl_t2(num(5),:)';A_gl_t2(num(6),:)';A_gl_t1(num(7),:)';A_gl_t2(num(8),:)'];        %对应的8个积分点PE值
    CZ=S*C;
    D(k,:)=subs(CZ,{a,b,c},{0,0,0});
   ebsino_L=[D(k,1),D(j,4),D(k,5);D(k,4),D(k,2),D(k,6);D(k,5),D(k,6),D(k,3)];
   ebsino_G=R_gl*ebsino_L*R_gl';
   DDD(k,:)=[ebsino_G(1,1),ebsino_G(2,2),ebsino_G(3,3),ebsino_G(1,2),ebsino_G(1,3),ebsino_G(2,3)];
    fprintf(fid_pe_p1,format_pe_p1,k+j,DDD(k,:));
end
%---------------------------------------------5.3全局模型焊接稳态区域写入

for jk=1:length(glob_local_wen)/8
     aaa=glob_local_wen((jk-1)*8+1,1)
    fprintf(fid_elset_p1,format_elset_p1,jk+j+k,aaa);

    num=(1:8)+(jk-1)*8;                      %对应的8个积分点编号
    C=[A_gl_s(num(1),:)';A_gl_s(num(2),:)';A_gl_s(num(3),:)';A_gl_s(num(4),:)';A_gl_s(num(5),:)';A_gl_s(num(6),:)';A_gl_s(num(7),:)';A_gl_s(num(8),:)'];        %对应的8个积分点PE值
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



