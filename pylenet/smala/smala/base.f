c file: bcs.f

c     Compute y = 1 if a < x < b, 0 otherwise
      subroutine dclip(x,y,a,b,n)
      real x(n),y(n),a,b
      integer n,i
cf2py intent(in) :: x,a,b,y,n
      do i=1,n
        if (x(i) < a .or. x(i) > b) then
          y(i) = 0.0
        else
          y(i) = 1.0
        end if
      end do
      end

c     Compute y = y + a * x
      subroutine adda(x,y,n,a)
      real x(n),y(n),a
      integer n,i
cf2py intent(inplace) :: y
cf2py intent(in) :: x,n,a
      forall(i=1:n) y(i) = y(i) + a * x(i)
      end

c     Compute y = y + w * x
      subroutine addamul211(w,x,nx,y,ny,a)
      real w(ny,nx),x(nx),y(ny),a,xi
      integer nx,ny,i,j
cf2py intent(inplace) :: y
cf2py intent(in) :: w,x,nx,ny,a
      do i=1,nx
        xi = a * x(i)
        forall(j=1:ny) y(j) = y(j) + w(j,i) * xi
      end do
      end

c     Compute y = y + wt * x
      subroutine addamul2t11(w,x,nx,y,ny,a)
      real w(nx,ny),x(nx),y(ny),a,xi
      integer nx,ny,i,j
cf2py intent(inplace) :: y
cf2py intent(in) :: w,x,nx,ny,a
      do i=1,nx
        xi = a * x(i)
        forall(j=1:ny) y(j) = y(j) + w(i,j) * xi
      end do
      end

c     Compute z = z + x * yt
      subroutine addamul11t2(x,nx,y,ny,z,a)
      real x(nx),y(ny),z(nx,ny),a
      integer nx,ny,i,j
cf2py intent(inplace) :: z
cf2py intent(in) :: x,y,nx,ny,a
      forall(i=1:nx,j=1:ny) z(i,j) = z(i,j) + a * x(i) * y(j)
      end

c     Compute a 2d downsampling: y = y + a*dspl(x)
      subroutine addadspl2(x,wx,hx,y,wy,hy,a,ws,hs)
      integer ws,hs,wx,hx,wy,hy,p,q
      real x(wx,hx),y(wy,hy),a
cf2py intent(inplace) :: y
cf2py intent(in) :: x,a,ws,hs,wx,hx,wy,hy
      do i=1,wy
        do j=1,hy
          p = ws*(i-1)
          q = hs*(j-1)
          do k=1,ws
            do l=1,hs
              y(i,j) = y(i,j) + a * x(p+k,q+l)
            end do
          end do
        end do
      end do
      end

c     Compute a 2d upsampling: y = y + a*uspl(x)
      subroutine addauspl2(x,wx,hx,y,wy,hy,a,ws,hs)
      integer ws,hs,wx,hx,wy,hy,p,q
      real x(wx,hx),y(wy,hy),a
cf2py intent(inplace) :: y
cf2py intent(in) :: x,a,ws,hs,wx,hx,wy,hy
      do i=1,wx
        do j=1,hx
          p = ws*(i-1)
          q = hs*(j-1)
          forall(k=1:ws,l=1:hs) y(p+k,q+l) = y(p+k,q+l) + a*x(i,j)
        end do
      end do
      end
      
c     Compute a 2d inner convolution: z = z + a*iconv(x,y)
      subroutine addaiconv2(x,y,z,wx,hx,wy,hy,wz,hz,a)
      integer wx,hx,wy,hy,wz,hz,i,j,k,l
      real x(wx,hx),y(wy,hy),z(wz,hz),a
cf2py intent(inplace) :: z
cf2py intent(in) :: x,y,wx,hx,wy,hy,wz,hz,a
      do k=1,wy
        do l=1,hy
          ykl = a*y(wy+1-k,hy+1-l)
          forall(i=1:wz,j=1:hz) z(i,j)=z(i,j)+x(i-1+k,j-1+l)*ykl
        end do
      end do
      end

c     Compute a 2d inner correlation: z = z + a*icorr(x,y)
      subroutine addaicorr2(x,y,z,wx,hx,wy,hy,wz,hz,a)
      integer wx,hx,wy,hy,wz,hz,i,j,k,l
      real x(wx,hx),y(wy,hy),z(wz,hz),a,ykl
cf2py intent(inplace) :: z  
cf2py intent(in) :: x,y,wx,hx,wy,hy,wz,hz,a    
      do k=1,wy
        do l=1,hy
          ykl=a*y(k,l)
          forall(i=1:wz,j=1:hz) z(i,j)=z(i,j)+x(i-1+k,j-1+l)*ykl
        end do
      end do
      end
      
c     Compute a 2d outer convolution: z = z + a*oconv(x,y)
      subroutine addaoconv2(x,y,z,wx,hx,wy,hy,wz,hz,a)
      integer wx,hx,wy,hy,wz,hz,i,j,k,l
      real x(wx,hx),y(wy,hy),z(wz,hz),a,ykl
cf2py intent(inplace) :: z
cf2py intent(in) :: x,y,wx,hx,wy,hy,wz,hz
      do k=1,wy
        do l=1,hy
          ykl = a*y(wy+1-k,hy+1-l)
          forall(i=1:wx,j=1:hx) z(i-1+k,j-1+l)=z(i-1+k,j-1+l)+ykl*x(i,j)
        end do
      end do
      end

