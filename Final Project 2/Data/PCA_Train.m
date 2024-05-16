function [Row_FACE_Data]=PCA_Train

people=40;

withinsample=5;

principlenum=50;

Row_FACE_Data=[];

for k=1:1: people
     % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     for m=1:2:10
         PathString=['orl3232' '\' num2str(k) '\' num2str(m) '.bmp'];
         ImageData=imread(PathString);
         ImageData=double(ImageData);
         if (k==1 && m==1)
            [row,col]=size(ImageData);
         end         
         
         RowConcatenate=[];
         %--arrange the image into a vector
         for n=1:row
             RowConcatenate=[RowConcatenate, ImageData(n,:)];
         end
         
         Row_FACE_Data=[Row_FACE_Data; RowConcatenate]; %
     end 
     % !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 
end % end of k=1:1:people