      subroutine mandel(n, image, max_iter)

      integer   ( kind = 4 ) :: n
      integer   ( kind = 4 ) :: image(n,n) 
      integer   ( kind = 4 ) :: max_iter
      integer   ( kind = 4 ) :: i, j, k
      real      ( kind = 8 ) :: x, x1, x2
      real      ( kind = 8 ) :: y, y1, y2
      real      ( kind = 8 ) :: x_max =   1.25D+00
      real      ( kind = 8 ) :: x_min = - 2.25D+00
      real      ( kind = 8 ) :: y_max =   1.75D+00
      real      ( kind = 8 ) :: y_min = - 1.75D+00

      do i = 1, n
        do j = 1, n

          x = (  real (     j - 1, kind = 8 ) * x_max   &
               + real ( n - j,     kind = 8 ) * x_min ) &
               / real ( n     - 1, kind = 8 )

          y = (  real (     i - 1, kind = 8 ) * y_max   &
               + real ( n - i,     kind = 8 ) * y_min ) &
               / real ( n     - 1, kind = 8 )

          image(i,j) = 0

          x1 = x
          y1 = y

          do k = 1, max_iter

             x2 = x1 * x1 - y1 * y1 + x
             y2 =  2 * x1 * y1 + y

             if ( x2 < -2.0D+00 .or. &
                   2.0D+00 < x2 .or. &
                  y2 < -2.0D+00 .or. &
                    2.0D+00 < y2 ) then

                    image(i,j) = k
                    exit

             end if

             x1 = x2
             y1 = y2

          end do

        end do
      end do

      end subroutine
